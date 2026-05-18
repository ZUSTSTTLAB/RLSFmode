import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import defaultdict, Counter
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
import sys
from datetime import datetime
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import math


class Logger(object):
    def __init__(self, filename="training_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class MultiTaskConfig:
    def __init__(self, task_type="pose"):
        self.task_type = task_type
        self.data_dir = ".."
        self.batch_size = 256
        self.val_batch_size = 32
        self.num_workers = 8

        if task_type == "pose":
            # 姿态识别配置 (分类任务)
            self.lr = 1e-3
            self.weight_decay = 1e-4
            self.epochs = 100
            self.pos_weight = 2.8
            self.descriptor_dim = 4  # 4维相互作用评分
        elif task_type == "affinity":
            # 亲和力预测配置 (回归任务)
            self.lr = 5e-3
            self.weight_decay = 1e-4
            self.epochs = 100
            self.descriptor_dim = 768  # 768维序列描述符


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
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return F.relu(out, inplace=True)


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


class PoseRecognitionBranch(nn.Module):
    """姿态识别任务的分子描述符分支 - 处理4维相互作用评分"""

    def __init__(self):
        super(PoseRecognitionBranch, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.LayerNorm(256)
        )

    def forward(self, x):
        return self.fc(x)


class AffinityPredictionBranch(nn.Module):
    """亲和力预测任务的分子描述符分支 - 处理768维序列描述符"""

    def __init__(self):
        super(AffinityPredictionBranch, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fc(x)


class MultiTaskResNet34(nn.Module):
    """多任务ResNet-34网络，同时支持姿态识别和亲和力预测"""

    def __init__(self, task_type="pose"):
        super(MultiTaskResNet34, self).__init__()
        self.task_type = task_type

        # 共享的CNN特征提取器
        self.conv1 = nn.Conv2d(10, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差层
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        # 注意力机制
        self.ca1 = ChannelAttention(64)
        self.ca2 = ChannelAttention(128)
        self.ca3 = ChannelAttention(256)
        self.ca4 = ChannelAttention(512)

        # 任务特定的分子描述符分支
        if task_type == "pose":
            self.descriptor_branch = PoseRecognitionBranch()
            self._init_pose_classifier()
        elif task_type == "affinity":
            self.descriptor_branch = AffinityPredictionBranch()
            self._init_affinity_regressor()

        # 共享的池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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

    def _init_pose_classifier(self):
        """初始化姿态识别分类器"""
        self.fusion_fc = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def _init_affinity_regressor(self):
        """初始化亲和力预测回归器"""
        self.fusion_fc = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )

        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x, descriptor):
        # 共享的CNN特征提取
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

        # 提取CNN特征
        cnn_feat = self.avgpool(x).flatten(1)

        # 任务特定的描述符处理
        descriptor_feat = self.descriptor_branch(descriptor)

        # 任务特定的融合和输出
        if self.task_type == "pose":
            combined = torch.cat([cnn_feat, descriptor_feat], dim=1)
            output = self.fusion_fc(combined)
            return output.squeeze()

        elif self.task_type == "affinity":
            combined = torch.cat([cnn_feat, descriptor_feat], dim=1)
            fused_feat = self.fusion_fc(combined)
            affinity = self.regressor(fused_feat)
            return affinity.squeeze()


class MultiTaskDataset(Dataset):
    def __init__(self, root_dir, mode="train", task_type="pose"):
        self.samples = []
        self.group_data = defaultdict(list)
        self.mode = mode
        self.task_type = task_type
        self.descriptor_dir = os.path.join(root_dir, "DESCRIPTORS")

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]) if mode == "train" else None

        # 根据任务类型加载数据
        if task_type == "pose":
            # 姿态识别：二分类数据
            for label in [0, 1]:
                cls_dir = os.path.join(root_dir, str(label))
                if not os.path.exists(cls_dir):
                    print(f"警告: 目录 {cls_dir} 不存在")
                    continue

                files = [f for f in os.listdir(cls_dir) if f.endswith('.npy')]
                for file in files:
                    self._add_sample(root_dir, file, label, task_type)
        else:
            # 亲和力预测：回归数据
            if not os.path.exists(root_dir):
                print(f"警告: 目录 {root_dir} 不存在")
                return

            files = [f for f in os.listdir(root_dir) if f.endswith('.npy')]
            for file in files:
                self._add_sample(root_dir, file, None, task_type)

        if task_type == "pose":
            self.class_counts = Counter([s['label'] for s in self.samples])
            print(f"\n{mode}集统计 - 负样本: {self.class_counts[0]}, 正样本: {self.class_counts[1]}")
        print(f"总样本数: {len(self.samples)}, 总组数: {len(self.group_data)}")

    def _add_sample(self, root_dir, file, label, task_type):
        """添加样本到数据集"""
        file_path = os.path.join(root_dir, str(label) if task_type == "pose" else root_dir, file)

        # 根据任务类型加载不同的描述符
        if task_type == "pose":
            descriptor_path = os.path.join(self.descriptor_dir, file.replace('.npy', '_interaction.txt'))
        else:
            descriptor_path = os.path.join(self.descriptor_dir, file.replace('.npy', '_sequence.txt'))

        group_name = file[:4]

        # 加载描述符特征
        descriptor_feature = self._load_descriptor(descriptor_path, task_type)

        self.samples.append({
            'path': file_path,
            'label': label,
            'group': group_name,
            'file': file,
            'descriptor': descriptor_feature
        })
        self.group_data[group_name].append(len(self.samples) - 1)

    def _load_descriptor(self, descriptor_path, task_type):
        """根据任务类型加载描述符"""
        default_value = np.zeros(4) if task_type == "pose" else np.zeros(768)

        if not os.path.exists(descriptor_path):
            print(f"警告: 描述符文件 {descriptor_path} 不存在")
            return default_value

        try:
            with open(descriptor_path, 'r') as f:
                values = [float(x) for x in f.read().strip().split()]

            if task_type == "pose":
                if len(values) == 4:
                    return np.array(values, dtype=np.float32)
                else:
                    print(f"错误: 期望4维特征，得到{len(values)}维")
                    return default_value
            else:
                if len(values) == 768:
                    return np.array(values, dtype=np.float32)
                else:
                    print(f"错误: 期望768维特征，得到{len(values)}维")
                    return default_value
        except Exception as e:
            print(f"无法读取描述符文件 {descriptor_path}: {str(e)}")
            return default_value

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            # 加载图像数据
            data = np.load(sample['path'])
            data = np.nan_to_num(data, nan=0.0)
            data = torch.FloatTensor(data).permute(2, 0, 1)
            data = (data - data.mean()) / (data.std() + 1e-8)

            if self.transform and self.mode == "train":
                data = self.transform(data)

            # 返回任务特定的数据
            return data, sample['label'], sample['group'], sample['file'], torch.FloatTensor(sample['descriptor'])

        except Exception as e:
            print(f"加载 {sample['path']} 出错: {str(e)}")
            return self.__getitem__(np.random.randint(0, len(self)))

