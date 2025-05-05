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
    def __init__(self, feature_dims=[64], classifier_dims=[32]):
        super().__init__()
        # 特征提取MLP (处理单位特征)
        input_dim = 9  # 单位ID(1) + 数量(1) + 属性(7)
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
        batch_features = []
        for units in batch:  # 遍历batch中的每个战斗实例
            instance_features = []
            for (unit_id, features) in units:  # 处理每个单位
                # 将ID作为额外特征
                unit_id_feature = torch.tensor([float(unit_id)], dtype=torch.float)
                combined = torch.cat([unit_id_feature, features])  # combined维度为 (10,)
                
                # 修复：调整输入维度以匹配 feature_mlp 的输入
                combined = combined[1:]  # 跳过 unit_id，只保留特征部分
                
                unit_feature = self.feature_mlp(combined)
                instance_features.append(unit_feature)
            
            # 聚合当前战斗的所有单位特征
            if len(instance_features) == 0:
                agg_features = torch.zeros(self.feature_mlp[-2].out_features)
            else:
                agg_features = torch.sum(torch.stack(instance_features), dim=0)
            batch_features.append(agg_features)
        
        return torch.stack(batch_features)

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
            unit_data = (
                torch.tensor(unit_id, dtype=torch.long),
                torch.tensor(features[1:], dtype=torch.float)  # 跳过ID，因为ID会单独处理
            )
            left_units.append(unit_data)
        
        # 处理右方单位
        right_units = []
        for unit_key, count in battle['right'].items():
            unit_id = int(unit_key.split('_')[1])
            attrs = self.monster_data[unit_id]
            # 归一化特征
            features = np.array([float(unit_id), float(count)] + attrs)
            features = self.scaler.transform(features.reshape(1, -1)).flatten()
            unit_data = (
                torch.tensor(unit_id, dtype=torch.long),
                torch.tensor(features[1:], dtype=torch.float)  # 跳过ID，因为ID会单独处理
            )
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

def load_data_from_results_folder(folder_path):
    data_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data_list.append(data)
    return data_list

# 使用示例
if __name__ == "__main__":
    # 加载数据
    data_list = load_data_from_results_folder("results")

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
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 初始化模型
    model = BattleModel()

    # 训练循环
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_losses = []
    test_accuracies = []

    for epoch in range(100):
        epoch_loss = 0  # 用于累积每个 batch 的损失
        batch_count = 0  # 记录 batch 的数量
        
        # 训练阶段
        model.train()
        for batch in train_dataloader:
            outputs = model(batch['left'], batch['right'])
            loss = F.binary_cross_entropy(outputs, batch['label'])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 累积损失和计数
            epoch_loss += loss.item()
            batch_count += 1
        
        # 计算每个 epoch 的平均损失
        avg_loss = epoch_loss / batch_count
        train_losses.append(avg_loss)
        
        # 测试阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_dataloader:
                outputs = model(batch['left'], batch['right'])
                predictions = (outputs > 0.5).float()  # 将概率转换为0或1
                correct += (predictions == batch['label']).sum().item()
                total += batch['label'].size(0)
        
        accuracy = correct / total
        test_accuracies.append(accuracy)
        
        # 每 10 个 epoch 输出一次结果
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # 可视化损失和准确率
    plt.figure(figsize=(12, 5))

    # 绘制训练损失
    plt.subplot(1, 2, 1)
    plt.plot(range(100), train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # 绘制测试准确率
    plt.subplot(1, 2, 2)
    plt.plot(range(100), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()