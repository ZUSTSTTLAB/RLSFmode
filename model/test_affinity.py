import os
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import argparse
from datetime import datetime


class Config:
    def __init__(self):
        self.batch_size = 32
        self.num_workers = 4
        self.rna_embed_dim = 768
        self.rna_feat_dim = 128
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = nn.functional.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return nn.functional.relu(out, inplace=True)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return out.view(b, c, 1, 1)


class ResNet34_10C(nn.Module):
    def __init__(self, config):
        super(ResNet34_10C, self).__init__()
        self.config = config

        self.conv1 = nn.Conv2d(10, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        self.ca1 = ChannelAttention(64)
        self.ca2 = ChannelAttention(128)
        self.ca3 = ChannelAttention(256)
        self.ca4 = ChannelAttention(512)

        self.rna_branch = nn.Sequential(
            nn.Linear(config.rna_embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, config.rna_feat_dim),
            nn.BatchNorm1d(config.rna_feat_dim),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        fused_dim = 512 + config.rna_feat_dim

        self.hierarchical_fusion = nn.Sequential(
            nn.Linear(fused_dim, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        self.regression_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, rna_embed):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.ca1(x) * x
        x = self.layer2(x)
        x = self.ca2(x) * x
        x = self.layer3(x)
        x = self.ca3(x) * x
        x = self.layer4(x)
        x = self.ca4(x) * x

        conv_feat = self.avgpool(x).flatten(1)
        rna_feat = self.rna_branch(rna_embed)
        combined = torch.cat([conv_feat, rna_feat], dim=1)
        latent = self.hierarchical_fusion(combined)
        affinity = self.regression_head(latent)

        return affinity


class TestDataset(Dataset):
    def __init__(self, poseaf_dir, rna_dir):
        """
        Args:
            poseaf_dir: 测试集PoseAF特征目录（af3文件夹）
            rna_dir: RNA嵌入目录（affinity_768文件夹）
        """
        self.poseaf_dir = poseaf_dir
        self.rna_dir = rna_dir
        self.samples = []

        # 获取所有PoseAF文件
        npy_files = [f for f in os.listdir(poseaf_dir) if f.endswith('.npy')]

        # 获取所有RNA文件（用于快速匹配）
        rna_files = {}
        for f in os.listdir(rna_dir):
            if f.endswith('.txt'):
                # 提取文件名（不含扩展名）
                base_name = f.replace('.txt', '')
                # 同时保存小写版本用于不区分大小写匹配
                rna_files[base_name.lower()] = os.path.join(rna_dir, f)

        print(f"找到 {len(rna_files)} 个RNA嵌入文件")

        for npy_file in npy_files:
            sample_name = npy_file.replace('.npy', '')
            npy_path = os.path.join(poseaf_dir, npy_file)

            # 尝试匹配RNA文件（匹配前4个字符，不区分大小写）
            matched_rna_path = None
            matched_key = None

            # 方法1：完全匹配（不区分大小写）
            sample_name_lower = sample_name.lower()
            if sample_name_lower in rna_files:
                matched_rna_path = rna_files[sample_name_lower]
                matched_key = sample_name_lower
            else:
                # 方法2：匹配前4个字符
                prefix = sample_name[:4].lower()
                for rna_key, rna_path in rna_files.items():
                    if rna_key.startswith(prefix) or prefix.startswith(rna_key[:4]):
                        matched_rna_path = rna_path
                        matched_key = rna_key
                        break

                # 方法3：如果前4个字符匹配不到，尝试前5个字符
                if matched_rna_path is None and len(sample_name) >= 5:
                    prefix5 = sample_name[:5].lower()
                    for rna_key, rna_path in rna_files.items():
                        if rna_key.startswith(prefix5) or prefix5.startswith(rna_key[:5]):
                            matched_rna_path = rna_path
                            matched_key = rna_key
                            break

            if matched_rna_path is None:
                print(f"警告: 未找到样本 {sample_name} 的RNA嵌入文件，跳过")
                continue

            self.samples.append({
                'name': sample_name,
                'npy_path': npy_path,
                'rna_path': matched_rna_path,
                'matched_rna_key': matched_key
            })

        print(f"成功匹配 {len(self.samples)} 个测试样本")

    def _load_rna_embed(self, txt_path):
        """从txt文件加载768维RNA嵌入"""
        try:
            with open(txt_path, 'r') as f:
                content = f.read().strip()
                # 使用正则表达式提取所有数字
                values = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', content)
                embed = np.array([float(v) for v in values], dtype=np.float32)

                if len(embed) != 768:
                    if len(embed) < 768:
                        embed = np.pad(embed, (0, 768 - len(embed)))
                    else:
                        embed = embed[:768]
                return embed
        except Exception as e:
            print(f"读取RNA嵌入失败 {txt_path}: {e}")
            return np.zeros(768, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            # 加载PoseAF特征
            data = np.load(sample['npy_path'])
            data = np.nan_to_num(data, nan=0.0)

            # 转换为 (C, H, W) 格式
            if len(data.shape) == 3:
                if data.shape[-1] == 10:
                    data = data.transpose(2, 0, 1)
                elif data.shape[0] != 10:
                    print(f"警告: {sample['name']} 形状异常: {data.shape}")
                    data = np.zeros((10, 20, 20), dtype=np.float32)
            else:
                print(f"警告: {sample['name']} 维度异常: {data.shape}")
                data = np.zeros((10, 20, 20), dtype=np.float32)

            # 确保通道数为10
            if data.shape[0] != 10:
                print(f"错误: {sample['name']} 通道数不是10: {data.shape[0]}")
                data = np.zeros((10, 20, 20), dtype=np.float32)

            # 标准化
            data = torch.FloatTensor(data)
            for c in range(data.shape[0]):
                channel_data = data[c]
                if channel_data.std() > 1e-8:
                    data[c] = (channel_data - channel_data.mean()) / (channel_data.std() + 1e-8)
                else:
                    data[c] = channel_data - channel_data.mean()

            # 加载RNA嵌入
            rna_embed = self._load_rna_embed(sample['rna_path'])
            rna_embed = torch.FloatTensor(rna_embed)

            return data, rna_embed, sample['name']

        except Exception as e:
            print(f"加载样本 {sample['name']} 失败: {e}")
            return torch.zeros(10, 20, 20), torch.zeros(768), sample['name']


class Predictor:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"\n使用设备: {self.device}")
        print(f"加载模型: {model_path}")

        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # 处理不同的checkpoint格式
        if isinstance(checkpoint, dict):
            if 'config' in checkpoint:
                config = checkpoint['config']
                print("使用checkpoint中的配置")
            else:
                config = Config()
                print("使用默认配置")

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            config = Config()
            state_dict = checkpoint

        # 确保配置属性存在
        if not hasattr(config, 'rna_embed_dim'):
            config.rna_embed_dim = 768
        if not hasattr(config, 'rna_feat_dim'):
            config.rna_feat_dim = 128

        print(f"模型配置: rna_embed_dim={config.rna_embed_dim}, rna_feat_dim={config.rna_feat_dim}")

        # 创建模型
        self.model = ResNet34_10C(config).to(self.device)

        # 加载权重
        try:
            self.model.load_state_dict(state_dict, strict=True)
            print("模型权重加载成功（严格模式）")
        except Exception as e:
            print(f"严格加载失败: {e}")
            print("尝试非严格加载...")
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"缺少的键: {missing[:5]}...")
            if unexpected:
                print(f"多余的键: {unexpected[:5]}...")
            print("模型权重加载成功（非严格模式）")

        self.model.eval()
        print("模型已设置为评估模式")

    def predict_batch(self, dataloader):
        """批量预测"""
        results = []

        with torch.no_grad():
            for data, rna_embed, names in tqdm(dataloader, desc="预测中"):
                data = data.to(self.device)
                rna_embed = rna_embed.to(self.device)

                outputs = self.model(data, rna_embed)
                affinities = outputs.cpu().numpy().flatten()

                for i in range(len(affinities)):
                    results.append({
                        'file_name': names[i],
                        'pred_affinity': float(affinities[i])
                    })

        return results

    def save_results(self, results, output_dir):
        """保存预测结果"""
        os.makedirs(output_dir, exist_ok=True)

        # 创建DataFrame
        df = pd.DataFrame(results)

        # 按预测亲和力降序排序
        df = df.sort_values('pred_affinity', ascending=False)

        # 保存CSV
        csv_path = os.path.join(output_dir, 'test_predictions.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n预测结果已保存: {csv_path}")

        # 生成详细报告
        report_path = os.path.join(output_dir, 'prediction_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("外部测试集预测报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总样本数: {len(results)}\n")
            f.write("\n注意: pred_affinity是归一化的结合亲和力\n")
            f.write("      值域范围 [0,1]，值越大表示结合亲和力越强\n")
            f.write("=" * 80 + "\n\n")

            # 统计信息
            affinities = [r['pred_affinity'] for r in results]
            f.write("预测结果统计:\n")
            f.write(f"  均值: {np.mean(affinities):.6f}\n")
            f.write(f"  标准差: {np.std(affinities):.6f}\n")
            f.write(f"  最小值: {np.min(affinities):.6f}\n")
            f.write(f"  最大值: {np.max(affinities):.6f}\n")
            f.write(f"  中位数: {np.median(affinities):.6f}\n\n")

            # Top预测结果
            f.write("=" * 80 + "\n")
            f.write("Top 20 高结合亲和力预测\n")
            f.write("=" * 80 + "\n")
            for i, row in df.head(20).iterrows():
                f.write(f"{i + 1:3d}. {row['file_name']:50s} affinity = {row['pred_affinity']:.6f}\n")

            # 预测值分布
            f.write("\n" + "=" * 80 + "\n")
            f.write("预测值分布\n")
            f.write("=" * 80 + "\n")
            bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
            for low, high in bins:
                count = sum(1 for a in affinities if low <= a < high)
                pct = count / len(affinities) * 100
                f.write(f"  [{low:.1f}, {high:.1f}): {count:4d} 样本 ({pct:5.1f}%)\n")

        print(f"预测报告已保存: {report_path}")

        # 保存简单格式的结果（一行一个样本）
        simple_path = os.path.join(output_dir, 'predictions_simple.txt')
        with open(simple_path, 'w', encoding='utf-8') as f:
            f.write("# file_name\tpred_affinity\n")
            for row in df.to_dict('records'):
                f.write(f"{row['file_name']}\t{row['pred_affinity']:.6f}\n")

        print(f"简单格式结果已保存: {simple_path}")


def main():
    parser = argparse.ArgumentParser(description='使用训练好的模型预测外部测试集')
    parser.add_argument('--model_path', type=str, default='affinity_model.pth', help='训练好的模型权重文件路径 (.pth)')
    parser.add_argument('--poseaf_dir', type=str, default='af3', help='测试集PoseAF特征文件夹路径')
    parser.add_argument('--rna_dir', type=str, default='affinity_768', help='RNA嵌入文件夹路径')
    parser.add_argument('--output_dir', type=str, default='test_predictions', help='输出结果文件夹')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')

    args = parser.parse_args()

    print("=" * 80)
    print("外部测试集预测")
    print("=" * 80)
    print(f"模型文件: {args.model_path}")
    print(f"测试集PoseAF目录: {args.poseaf_dir}")
    print(f"RNA嵌入目录: {args.rna_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"批大小: {args.batch_size}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 检查目录
    if not os.path.exists(args.poseaf_dir):
        print(f"错误: PoseAF目录不存在: {args.poseaf_dir}")
        return
    if not os.path.exists(args.rna_dir):
        print(f"错误: RNA嵌入目录不存在: {args.rna_dir}")
        return
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return

    # 加载测试数据集
    print("\n加载测试数据...")
    test_dataset = TestDataset(args.poseaf_dir, args.rna_dir)

    if len(test_dataset) == 0:
        print("错误: 没有找到有效的测试样本")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 初始化预测器
    print("\n初始化模型...")
    predictor = Predictor(args.model_path)

    # 执行预测
    print("\n开始预测...")
    results = predictor.predict_batch(test_loader)

    # 保存结果
    print("\n保存结果...")
    predictor.save_results(results, args.output_dir)

    # 打印部分结果示例
    print("\n" + "=" * 80)
    print("预测结果示例（前10个）")
    print("=" * 80)
    df = pd.DataFrame(results).sort_values('pred_affinity', ascending=False)
    for i, row in df.head(10).iterrows():
        print(f"{i + 1:3d}. {row['file_name']:50s} -> {row['pred_affinity']:.6f}")

    print("\n" + "=" * 80)
    print(f"预测完成！结果保存在: {args.output_dir}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()