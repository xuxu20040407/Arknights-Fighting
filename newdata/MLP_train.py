import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class BattleModel(nn.Module):
    def __init__(self, feature_dims=[16,16,32], classifier_dims=[32,16,8]):
        super().__init__()
        # 特征提取MLP (处理单位特征)
        input_dim = 10  # 单位ID(1) + 数量(1) + 属性(8)
        feature_layers = []
        prev_dim = input_dim
        for dim in feature_dims:
            feature_layers.append(nn.Linear(prev_dim, dim))
            feature_layers.append(nn.ReLU())
            prev_dim = dim
        self.feature_mlp = nn.Sequential(*feature_layers)
        
        # 分类器MLP (处理聚合后的双方特征)
        classifier_input = feature_dims[-1] * 2
        classifier_layers = []
        prev_dim = classifier_input
        for dim in classifier_dims:
            classifier_layers.append(nn.Linear(prev_dim, dim))
            classifier_layers.append(nn.ReLU())
            prev_dim = dim
        classifier_layers.append(nn.Linear(prev_dim, 1))  # 最终输出层
        self.classifier = nn.Sequential(*classifier_layers)
        
    def forward(self, left_batch, right_batch):
        # 处理左右双方的单位数据
        left_features = self.process_batch(left_batch)
        right_features = self.process_batch(right_batch)
        
        # 拼接双方特征并分类
        combined = torch.cat([left_features, right_features], dim=1)
        logits = self.classifier(combined)
        return torch.sigmoid(logits.squeeze())
        
    def process_batch(self, batch):
        # batch 是 List[Tensor]，每个 Tensor 的形状是 (num_units, feature_dim)
        batch_features = []
        for units in batch:  # units 是 Tensor，形状为 (num_units, feature_dim)
            if units.numel() == 0:
                # 处理空战斗实例
                agg_features = torch.zeros(self.feature_mlp[-2].out_features, 
                                        device=units.device)
            else:
                # 直接通过特征提取MLP
                unit_features = self.feature_mlp(units)  # (num_units, feature_dim)
                # 聚合特征（求和）
                agg_features = torch.sum(unit_features, dim=0)
            batch_features.append(agg_features)
        
        return torch.stack(batch_features)  # (batch_size, feature_dim)

class BattleDataset(Dataset):
    def __init__(self, data_list, monster_csv_path):
        self.data_list = data_list
        self.monster_data = self.load_monster_data(monster_csv_path)
        self.scaler = StandardScaler()
        
        # 收集所有单位ID和属性用于归一化
        all_features = []
        for data in data_list:
            for side in ['left', 'right']:
                for unit_key, count in data[side].items():
                    unit_id = int(unit_key.split('_')[1])
                    attrs = self.monster_data[unit_id]
                    all_features.append([float(unit_id), float(count)] + attrs)
        
        # 训练归一化器
        if all_features:
            self.scaler.fit(all_features)
        
    @staticmethod
    def load_monster_data(csv_path):
        data = {}
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                unit_id = int(row['\ufeffID'])  # 处理BOM字符
                attrs = [
                    float(row['生命值'] or 0),
                    float(row['攻击力'] or 0),
                    float(row['防御力'] or 0),
                    float(row['法术抗性'] or 0),
                    float(row['攻击间隔'] or 0),
                    float(row['移动速度'] or 0),
                    float(row['攻击范围半径'] or 0),
                    float(row['法伤'] or 0)
                ]
                data[unit_id] = attrs
        return data
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        battle = self.data_list[idx]
        
        # 处理左方单位
        left_units = []
        for unit_key, count in battle['left'].items():
            unit_id = int(unit_key.split('_')[1])
            attrs = self.monster_data[unit_id]
            # 归一化特征
            features = np.array([float(unit_id), float(count)] + attrs)
            features = self.scaler.transform(features.reshape(1, -1)).flatten()
            unit_data = torch.tensor(features, dtype=torch.float)
            left_units.append(unit_data)
        
        # 处理右方单位
        right_units = []
        for unit_key, count in battle['right'].items():
            unit_id = int(unit_key.split('_')[1])
            attrs = self.monster_data[unit_id]
            # 归一化特征
            features = np.array([float(unit_id), float(count)] + attrs)
            features = self.scaler.transform(features.reshape(1, -1)).flatten()
            unit_data = torch.tensor(features, dtype=torch.float)
            right_units.append(unit_data)
        
        label = 1 if battle['wl'] == 'left' else 0
        return {
            'left': left_units,
            'right': right_units,
            'label': torch.tensor(label, dtype=torch.float)
        }

def collate_fn(batch):
    return {
        'left': [item['left'] for item in batch],
        'right': [item['right'] for item in batch],
        'label': torch.stack([item['label'] for item in batch])
    }


# 使用示例
if __name__ == "__main__":
    # 加载数据
    with open(r'output.json', 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    # 划分训练集和测试集，9:1
    train_data, test_data = train_test_split(data_list, test_size=0.1, random_state=42)
    # 打印训练集和测试集的大小
    print(f"训练集大小: {len(train_data)}")
    print(f"测试集大小: {len(test_data)}")
    # 创建训练集和测试集的数据加载器
    train_dataset = BattleDataset(train_data, r'data\monster.csv')
    test_dataset = BattleDataset(test_data, r'data\monster.csv')

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化模型并移动到设备
    model = BattleModel().to(device)

    # 训练循环
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # 每10个epoch学习率减半
    best_model_path = "best_model_mlp.pth"
    best_test_accuracy = 0.0

    train_accuracies = []
    test_accuracies = []

    for epoch in range(100):
        epoch_loss = 0  # 用于累积每个 batch 的损失
        batch_count = 0  # 记录 batch 的数量
        train_correct = 0
        train_total = 0

        # 训练阶段
        model.train()
        for batch in train_dataloader:
            # 将数据移动到设备
            left = [torch.stack(units).to(device) for units in batch['left']]
            right = [torch.stack(units).to(device) for units in batch['right']]
            labels = batch['label'].to(device)
            
            outputs = model(left, right)
            loss = F.binary_cross_entropy(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 累积损失和计数
            epoch_loss += loss.item()
            batch_count += 1

            # 计算训练准确率
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
        
        train_accuracy = train_correct / train_total
        train_accuracies.append(train_accuracy)
        
        # 测试阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_dataloader:
                # 将数据移动到设备
                left = [torch.stack(units).to(device) for units in batch['left']]
                right = [torch.stack(units).to(device) for units in batch['right']]
                labels = batch['label'].to(device)
                
                outputs = model(left, right)
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        test_accuracy = correct / total
        test_accuracies.append(test_accuracy)
        
        # 保存最佳模型
        if test_accuracy >= best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with accuracy: {best_test_accuracy:.4f}")
        
        # 更新学习率
        scheduler.step()
        
        # 每 3 个 epoch 输出一次结果
        if epoch % 3 == 0:
            print(f"Epoch {epoch}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))
    print(f"Loaded best model with accuracy: {best_test_accuracy:.4f}")

    # 可视化准确率
    plt.figure(figsize=(12, 5))

    # 绘制训练和测试准确率
    plt.plot(range(100), train_accuracies, label="Train Accuracy")
    plt.plot(range(100), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()