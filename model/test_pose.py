import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import argparse
from datetime import datetime


# 添加Config类（与训练时保持一致）
class Config:
    def __init__(self):
        self.data_dir = "."
        self.batch_size = 256
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.epochs = 100
        self.num_workers = 4
        self.fusion_type = "concat"
        self.save_criteria = "combined"
        self.patience = 30
        self.rdock_feature_dim = 4


# 定义模型结构（与训练时完全一致）
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

        self.rdock_fc = nn.Sequential(
            nn.Linear(config.rdock_feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.LayerNorm(256)
        )

        if self.config.fusion_type == "concat":
            self.fc_input_dim = 768
            self.rdock_adapter = None
            self.gate_fc = None
            self.attn_fc = None
        elif self.config.fusion_type == "gate":
            self.fc_input_dim = 512
            self.gate_fc = nn.Sequential(
                nn.Linear(512 + 512, 512),
                nn.Sigmoid()
            )
            self.rdock_adapter = nn.Sequential(
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.ReLU()
            )
            self.attn_fc = None
        elif self.config.fusion_type == "attention":
            self.fc_input_dim = 512
            self.attn_fc = nn.Sequential(
                nn.Linear(512 + 512, 256),
                nn.ReLU(),
                nn.Linear(256, 2),
                nn.Softmax(dim=1)
            )
            self.rdock_adapter = nn.Sequential(
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.ReLU()
            )
            self.gate_fc = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
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

    def forward(self, x, rdock):
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

        img_feat = self.avgpool(x).flatten(1)
        rdock_feat = self.rdock_fc(rdock)

        if self.config.fusion_type == "concat":
            combined = torch.cat([img_feat, rdock_feat], dim=1)
        elif self.config.fusion_type == "gate":
            adapted_rdock = self.rdock_adapter(rdock_feat)
            combined = torch.cat([img_feat, adapted_rdock], dim=1)
            gate = self.gate_fc(combined)
            combined = img_feat * gate + adapted_rdock * (1 - gate)
        elif self.config.fusion_type == "attention":
            adapted_rdock = self.rdock_adapter(rdock_feat)
            combined = torch.cat([img_feat, adapted_rdock], dim=1)
            attn_weights = self.attn_fc(combined)
            combined = img_feat * attn_weights[:, 0].unsqueeze(1) + \
                       adapted_rdock * attn_weights[:, 1].unsqueeze(1)

        return self.fc(combined)


class TestDataset(Dataset):
    def __init__(self, poseaf_dir, rdock_dir):
        self.samples = []

        npy_files = [f for f in os.listdir(poseaf_dir) if f.endswith('.npy')]

        for npy_file in npy_files:
            file_name = npy_file.replace('.npy', '')
            npy_path = os.path.join(poseaf_dir, npy_file)
            rdock_path = os.path.join(rdock_dir, f"{file_name}.txt")

            rdock_feature = np.zeros(4, dtype=np.float32)
            rdock_score = 0.0
            if os.path.exists(rdock_path):
                try:
                    with open(rdock_path, 'r') as f:
                        lines = f.readlines()
                        for i in range(min(4, len(lines))):
                            rdock_feature[i] = float(lines[i].strip())
                        rdock_score = rdock_feature[0]
                except Exception as e:
                    print(f"警告: 无法读取 {rdock_path}, 错误: {e}")

            group_name = file_name.split('_')[0] if '_' in file_name else file_name[:4]

            self.samples.append({
                'file_name': file_name,
                'npy_path': npy_path,
                'rdock_feature': rdock_feature,
                'rdock_score': rdock_score,
                'group': group_name
            })

        print(f"加载了 {len(self.samples)} 个测试样本")
        print(f"注意: RDOCK评分越低越好，排名时数值小的排前面")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            data = np.load(sample['npy_path'])
            data = np.nan_to_num(data, nan=0.0)
            data = torch.FloatTensor(data).permute(2, 0, 1)
            data = (data - data.mean()) / (data.std() + 1e-8)

            return {
                'data': data,
                'file_name': sample['file_name'],
                'rdock_feature': torch.FloatTensor(sample['rdock_feature']),
                'rdock_score': sample['rdock_score'],
                'group': sample['group']
            }
        except Exception as e:
            print(f"加载 {sample['npy_path']} 出错: {e}")
            return self.__getitem__(0)


class Tester:
    def __init__(self, model_path, fusion_type="concat", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device)

        print(f"\n检查模型文件格式...")
        if isinstance(checkpoint, dict):
            if 'config' in checkpoint:
                config = checkpoint['config']
                print(f"使用checkpoint中的配置: fusion_type={config.fusion_type}")
                if not hasattr(config, 'rdock_feature_dim'):
                    config.rdock_feature_dim = 4
            else:
                config = Config()
                config.fusion_type = fusion_type
                config.rdock_feature_dim = 4

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            config = Config()
            config.fusion_type = fusion_type
            config.rdock_feature_dim = 4
            state_dict = checkpoint

        print(f"\n创建模型: fusion_type={config.fusion_type}, rdock_feature_dim={config.rdock_feature_dim}")
        self.model = ResNet34_10C(config).to(self.device)

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
        print(f"\n模型加载完成，使用设备: {self.device}")

    def predict_batch(self, dataloader):
        """批量预测"""
        results = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="预测中"):
                data = batch['data'].to(self.device)
                rdock = batch['rdock_feature'].to(self.device)

                outputs = self.model(data, rdock)
                scores = outputs.cpu().numpy().flatten()

                for i in range(len(scores)):
                    results.append({
                        'file_name': batch['file_name'][i],
                        'pred_score': float(scores[i]),
                        'rdock_score': float(batch['rdock_score'][i]),
                        'group': batch['group'][i]
                    })

        return results

    def get_rankings(self, results):
        """按组计算排名（注意：RDOCK分数越小排名越高）"""
        group_results = defaultdict(list)

        for r in results:
            group_results[r['group']].append(r)

        rankings = {}
        for group, samples in group_results.items():
            # 模型预测分数：越大越好，降序排序
            sorted_by_pred = sorted(samples, key=lambda x: x['pred_score'], reverse=True)
            # RDOCK分数：越小越好，升序排序（数值小的排前面）
            sorted_by_rdock = sorted(samples, key=lambda x: x['rdock_score'])

            rankings[group] = {
                'top1_by_pred': sorted_by_pred[0] if sorted_by_pred else None,
                'top1_by_rdock': sorted_by_rdock[0] if sorted_by_rdock else None,
                'top3_by_pred': sorted_by_pred[:3],
                'top3_by_rdock': sorted_by_rdock[:3],
                'all_samples': sorted_by_pred
            }

        return rankings

    def save_results(self, results, rankings, output_dir):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)

        # 1. 保存所有样本的预测分数
        df_all = pd.DataFrame(results)
        df_all = df_all[['file_name', 'group', 'pred_score', 'rdock_score']]
        df_all.to_csv(os.path.join(output_dir, 'all_predictions.csv'), index=False)
        print(f"保存所有预测结果: all_predictions.csv")

        # 2. 保存每个组的排名结果
        with open(os.path.join(output_dir, 'group_rankings.txt'), 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("组内排名结果\n")
            f.write("=" * 80 + "\n")
            f.write("注意: 模型预测分数越高越好，RDOCK评分越低越好\n\n")

            for group in sorted(rankings.keys()):
                f.write(f"\n{'=' * 80}\n")
                f.write(f"组: {group}\n")
                f.write(f"{'=' * 80}\n")

                f.write(f"\n【模型预测 Top 1 (分数越高越好)】\n")
                top1_pred = rankings[group]['top1_by_pred']
                if top1_pred:
                    f.write(f"  {top1_pred['file_name']}: pred_score={top1_pred['pred_score']:.6f}, "
                            f"rdock_score={top1_pred['rdock_score']:.4f}\n")

                f.write(f"\n【RDOCK评分 Top 1 (分数越低越好)】\n")
                top1_rdock = rankings[group]['top1_by_rdock']
                if top1_rdock:
                    f.write(f"  {top1_rdock['file_name']}: rdock_score={top1_rdock['rdock_score']:.4f}, "
                            f"pred_score={top1_rdock['pred_score']:.6f}\n")

                f.write(f"\n【模型预测 Top 3】\n")
                for i, sample in enumerate(rankings[group]['top3_by_pred'], 1):
                    f.write(f"  {i}. {sample['file_name']}: pred_score={sample['pred_score']:.6f}, "
                            f"rdock_score={sample['rdock_score']:.4f}\n")

                f.write(f"\n【RDOCK评分 Top 3 (分数越低越好)】\n")
                for i, sample in enumerate(rankings[group]['top3_by_rdock'], 1):
                    f.write(f"  {i}. {sample['file_name']}: rdock_score={sample['rdock_score']:.4f}, "
                            f"pred_score={sample['pred_score']:.6f}\n")

        print(f"保存组排名结果: group_rankings.txt")

        # 3. 保存每个组的详细CSV
        for group, data in rankings.items():
            df_group = pd.DataFrame(data['all_samples'])
            df_group = df_group[['file_name', 'pred_score', 'rdock_score']]
            safe_group_name = group.replace('/', '_').replace('\\', '_')
            df_group.to_csv(os.path.join(output_dir, f'{safe_group_name}_rankings.csv'), index=False)

        print(f"保存各组详细结果: {len(rankings)} 个CSV文件")

        # 4. 保存统计摘要
        with open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("测试结果摘要\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"总样本数: {len(results)}\n")
            f.write(f"总组数: {len(rankings)}\n\n")

            pred_scores = [r['pred_score'] for r in results]
            f.write("预测分数统计 (越高越好):\n")
            f.write(f"  均值: {np.mean(pred_scores):.6f}\n")
            f.write(f"  标准差: {np.std(pred_scores):.6f}\n")
            f.write(f"  最小值: {np.min(pred_scores):.6f}\n")
            f.write(f"  最大值: {np.max(pred_scores):.6f}\n")
            f.write(f"  中位数: {np.median(pred_scores):.6f}\n\n")

            rdock_scores = [r['rdock_score'] for r in results]
            f.write("RDOCK分数统计 (越低越好):\n")
            f.write(f"  均值: {np.mean(rdock_scores):.6f}\n")
            f.write(f"  标准差: {np.std(rdock_scores):.6f}\n")
            f.write(f"  最小值: {np.min(rdock_scores):.6f}\n")
            f.write(f"  最大值: {np.max(rdock_scores):.6f}\n")
            f.write(f"  中位数: {np.median(rdock_scores):.6f}\n\n")

            f.write("各组样本数:\n")
            for group in sorted(rankings.keys()):
                f.write(f"  {group}: {len(rankings[group]['all_samples'])}\n")

        print(f"保存统计摘要: summary.txt")

    def print_group_top1(self, rankings):
        """打印每个组的Top1结果"""
        print("\n" + "=" * 80)
        print("各组排名第一的样本")
        print("=" * 80)
        print("注意: 模型预测分数越高越好，RDOCK评分越低越好")

        for group in sorted(rankings.keys()):
            print(f"\n{'=' * 60}")
            print(f"【组: {group}】")

            top1_pred = rankings[group]['top1_by_pred']
            if top1_pred:
                print(f"\n  模型预测第一名 (分数越高越好):")
                print(f"    样本名: {top1_pred['file_name']}")
                print(f"    预测分数: {top1_pred['pred_score']:.6f}")
                print(f"    RDOCK分数: {top1_pred['rdock_score']:.4f}")

            top1_rdock = rankings[group]['top1_by_rdock']
            if top1_rdock:
                print(f"\n  RDOCK评分第一名 (分数越低越好):")
                print(f"    样本名: {top1_rdock['file_name']}")
                print(f"    RDOCK分数: {top1_rdock['rdock_score']:.4f}")
                print(f"    预测分数: {top1_rdock['pred_score']:.6f}")

            # 如果是同一个样本，显示合并信息
            if top1_pred and top1_rdock and top1_pred['file_name'] == top1_rdock['file_name']:
                print(f"\n  ⭐ 模型预测和RDOCK评分第一名是同一个样本！")


def main():
    parser = argparse.ArgumentParser(description='测试模型并生成预测结果')
    parser.add_argument('--model_path', type=str, default='pose__model.pth', help='模型权重文件路径')
    parser.add_argument('--poseaf_dir', type=str, default='af3', help='poseaf特征文件夹路径')
    parser.add_argument('--rdock_dir', type=str, default='rdock2', help='RDOCK评分文件夹路径（4维特征）')
    parser.add_argument('--output_dir', type=str, default='test_results', help='输出结果文件夹')
    parser.add_argument('--fusion_type', type=str, default='concat', choices=['concat', 'gate', 'attention'],
                        help='融合方式，需与训练时一致')
    parser.add_argument('--batch_size', type=int, default=256, help='批处理大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')

    args = parser.parse_args()

    print("=" * 60)
    print("模型测试")
    print("=" * 60)
    print(f"模型文件: {args.model_path}")
    print(f"PoseAF目录: {args.poseaf_dir}")
    print(f"RDOCK目录: {args.rdock_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"融合方式: {args.fusion_type}")
    print(f"批大小: {args.batch_size}")
    print(f"注意: RDOCK评分越低越好，排名时数值小的排前面")
    print(f"测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if not os.path.exists(args.poseaf_dir):
        print(f"错误: poseaf目录不存在: {args.poseaf_dir}")
        return
    if not os.path.exists(args.rdock_dir):
        print(f"错误: RDOCK目录不存在: {args.rdock_dir}")
        return
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return

    print("\n加载测试数据...")
    test_dataset = TestDataset(args.poseaf_dir, args.rdock_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print("\n初始化模型...")
    tester = Tester(args.model_path, args.fusion_type)

    print("\n开始预测...")
    results = tester.predict_batch(test_loader)

    print("\n计算组内排名...")
    rankings = tester.get_rankings(results)

    # 打印各组Top1结果
    tester.print_group_top1(rankings)

    print("\n保存结果...")
    tester.save_results(results, rankings, args.output_dir)

    print(f"\n测试完成！结果保存在: {args.output_dir}")
    print(f"测试结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()