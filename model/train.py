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


class MultiTaskTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_train_loss = float('inf')
        self.best_val_loss = float('inf')
        self._init_datasets()
        self._init_model()

    def _init_datasets(self):
        print(f"\n初始化{self.config.task_type}任务数据集...")

        if self.config.task_type == "pose":
            # 姿态识别：训练集、验证集、测试集
            self.train_set = MultiTaskDataset(
                os.path.join(self.config.data_dir, "train"), "train", "pose"
            )
            self.val_set = MultiTaskDataset(
                os.path.join(self.config.data_dir, "val"), "val", "pose"
            )
            self.test_set = MultiTaskDataset(
                os.path.join(self.config.data_dir, "test"), "test", "pose"
            )

            # 为分类任务设置加权采样
            weights = [1.0 / self.train_set.class_counts[s['label']] for s in self.train_set.samples]
            self.sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

        else:  # affinity task
            # 亲和力预测：训练集、验证集、测试集
            self.train_set = MultiTaskDataset(
                os.path.join(self.config.data_dir, "train"), "train", "affinity"
            )
            self.val_set = MultiTaskDataset(
                os.path.join(self.config.data_dir, "val"), "val", "affinity"
            )
            self.test_set = MultiTaskDataset(
                os.path.join(self.config.data_dir, "test"), "test", "affinity"
            )
            self.sampler = None

        # 训练数据加载器
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.config.batch_size,
            sampler=self.sampler,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

        # 验证数据加载器
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

        # 测试数据加载器
        self.test_loader = DataLoader(
            self.test_set,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

    def _init_model(self):
        print(f"\n初始化{self.config.task_type}任务模型...")
        self.model = MultiTaskResNet34(self.config.task_type).to(self.device)

        if self.config.task_type == "pose":
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.epochs}")

        for inputs, labels, _, _, descriptors in progress_bar:
            inputs = inputs.to(self.device)
            labels = labels.float().to(self.device)
            descriptors = descriptors.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs, descriptors)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        self.scheduler.step()
        return total_loss / len(self.train_loader)

    def evaluate(self, data_loader):
        """在指定数据加载器上评估模型"""
        self.model.eval()
        all_outputs = []
        all_labels = []
        all_groups = []
        all_files = []
        total_loss = 0

        with torch.no_grad():
            for inputs, labels, groups, files, descriptors in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.float().to(self.device)
                descriptors = descriptors.to(self.device)

                outputs = self.model(inputs, descriptors)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                all_outputs.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_groups.extend(groups)
                all_files.extend(files)

        avg_loss = total_loss / len(data_loader)
        return avg_loss, all_outputs, all_labels, all_groups, all_files

    def calculate_pose_metrics(self, outputs, labels, groups):
        """计算姿态识别任务的指标"""
        probs = np.array(outputs)
        labels = np.array(labels)

        # AUC
        try:
            auc = roc_auc_score(labels, probs)
        except:
            auc = 0.5

        # Precision
        preds = (probs > 0.5).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        precision = tp / (tp + fp + 1e-8)

        # Top-1 和 Top-3 成功率
        group_results = defaultdict(list)
        for prob, label, group in zip(probs, labels, groups):
            group_results[group].append({'prob': prob, 'label': label})

        top1_success = 0
        top3_success = 0
        total_groups = len(group_results)

        for group, samples in group_results.items():
            sorted_samples = sorted(samples, key=lambda x: x['prob'], reverse=True)

            # Top-1
            if sorted_samples[0]['label'] == 1:
                top1_success += 1

            # Top-3
            if any(s['label'] == 1 for s in sorted_samples[:min(3, len(sorted_samples))]):
                top3_success += 1

        top1_acc = top1_success / total_groups if total_groups > 0 else 0
        top3_acc = top3_success / total_groups if total_groups > 0 else 0

        return {
            'auc': auc,
            'precision': precision,
            'top1': top1_acc,
            'top3': top3_acc
        }

    def calculate_affinity_metrics(self, outputs, labels):
        """计算亲和力预测任务的指标"""
        preds = np.array(outputs)
        labels = np.array(labels)

        mse = mean_squared_error(labels, preds)
        r2 = r2_score(labels, preds)

        return {
            'mse': mse,
            'r2': r2
        }

    def train_and_evaluate(self):
        """训练并在测试集上评估模型"""
        print(f"\n{'=' * 60}")
        print(f"{f' 开始{self.config.task_type}任务训练 ':=^60}")
        print(f"{'=' * 60}")
        print(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"设备: {self.device}")
        print(f"训练样本数: {len(self.train_set)}")
        print(f"验证样本数: {len(self.val_set)}")
        print(f"测试样本数: {len(self.test_set)}\n")

        best_model_path = f"best_{self.config.task_type}_model.pth"

        # 训练循环
        for epoch in range(1, self.config.epochs + 1):
            # 训练阶段
            train_loss = self.train_epoch(epoch)

            # 验证阶段
            val_loss, _, _, _, _ = self.evaluate(self.val_loader)

            print(f"\nEpoch {epoch}/{self.config.epochs}:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_loss:.4f}")

            # 保存最佳模型
            if self.config.task_type == "pose":
                # 姿态识别：保存训练集loss最低的模型
                if train_loss < self.best_train_loss:
                    self.best_train_loss = train_loss
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"  ★ 保存最佳模型 (训练损失: {train_loss:.4f})")
            else:
                # 亲和力预测：保存验证集loss最低的模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"  ★ 保存最佳模型 (验证损失: {val_loss:.4f})")

        # 最终测试评估
        self.final_test_evaluation(best_model_path)

    def final_test_evaluation(self, model_path):
        """在测试集上进行最终评估"""
        print(f"\n{'=' * 60}")
        print(f"{f' {self.config.task_type}任务测试集评估 ':=^60}")
        print(f"{'=' * 60}")

        if not os.path.exists(model_path):
            print(f"错误: 未找到最佳模型文件 {model_path}")
            return

        # 加载最佳模型
        self.model.load_state_dict(torch.load(model_path))
        print("已加载最佳模型进行测试集评估")

        # 在测试集上评估
        test_loss, test_outputs, test_labels, test_groups, test_files = self.evaluate(self.test_loader)

        print(f"\n测试集损失: {test_loss:.4f}")

        if self.config.task_type == "pose":
            # 姿态识别任务指标
            metrics = self.calculate_pose_metrics(test_outputs, test_labels, test_groups)

            print(f"\n★ 姿态识别测试集性能:")
            print(f"  AUC: {metrics['auc']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Top-1预测成功率: {metrics['top1']:.4f}")