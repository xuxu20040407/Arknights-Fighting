import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import dense_to_sparse
import numpy as np
import os
import json
from torch_geometric.loader import DataLoader
import csv

def load_monster_data(csv_path):
    monster_data = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            monster_id = int(row['\ufeffID'])  # 注意BOM字符
            # 将所有属性转换为float，空值设为0
            attributes = [
                float(row['生命值']) if row['生命值'] else 0,
                float(row['攻击力']) if row['攻击力'] else 0,
                float(row['防御力']) if row['防御力'] else 0,
                float(row['法术抗性']) if row['法术抗性'] else 0,
                float(row['攻击间隔']) if row['攻击间隔'] else 0,
                float(row['移动速度']) if row['移动速度'] else 0,
                float(row['攻击范围半径']) if row['攻击范围半径'] else 0,
                float(row['法伤']) if row['法伤'] else 0
            ]
            monster_data[monster_id] = attributes
    return monster_data

def load_data_from_results_folder(folder_path):
    data_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data_list.append(data)
    return data_list

class BattleDataset:
    def __init__(self, data_list, monster_csv_path):
        self.data_list = data_list
        self.monster_data = load_monster_data(monster_csv_path)
        
        # 收集所有单位ID确定嵌入层大小
        self.all_unit_ids = set()
        for data in data_list:
            self.all_unit_ids.update(int(unit.split('_')[1]) for unit in data['left'].keys())
            self.all_unit_ids.update(int(unit.split('_')[1]) for unit in data['right'].keys())
        self.max_unit_id = max(self.all_unit_ids) if self.all_unit_ids else 0
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        battle = self.data_list[idx]
        
        # 构建节点特征（添加阵营标识）
        node_features = []
        left_indices = []  # 存储左方节点索引
        right_indices = []  # 存储右方节点索引
        
        # 处理左方单位
        for unit, count in battle['left'].items():
            unit_id = int(unit.split('_')[1])
            monster_info = self.monster_data.get(unit_id, [0]*8)
            # 特征: [unit_id, count, 阵营(左=1), 属性1, 属性2...]
            node_features.append([unit_id, float(count), 1.0] + monster_info)
            left_indices.append(len(node_features)-1)
            
        # 处理右方单位
        for unit, count in battle['right'].items():
            unit_id = int(unit.split('_')[1])
            monster_info = self.monster_data.get(unit_id, [0]*8)
            # 特征: [unit_id, count, 阵营(右=0), 属性1, 属性2...]
            node_features.append([unit_id, float(count), 0.0] + monster_info)
            right_indices.append(len(node_features)-1)
        
        # 构建三种边类型（使用掩码矩阵）
        num_nodes = len(node_features)
        edge_mask = torch.zeros(num_nodes, num_nodes, dtype=torch.long)
        
        # 左方内部全连接（类型1）
        for i in left_indices:
            for j in left_indices:
                if i != j:
                    edge_mask[i, j] = 1
                    
        # 右方内部全连接（类型2）
        for i in right_indices:
            for j in right_indices:
                if i != j:
                    edge_mask[i, j] = 2
                    
        # 跨阵营全连接（类型3）
        for i in left_indices:
            for j in right_indices:
                edge_mask[i, j] = 3
                edge_mask[j, i] = 3
                
        edge_index, edge_type = dense_to_sparse(edge_mask)
        
        # 标签
        label = 1 if battle['wl'] == 'left' else 0
        
        return Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            edge_type=edge_type,  # 边类型1-3
            y=torch.tensor([[label]], dtype=torch.float)
        )

from torch_geometric.nn import GINEConv

class BattleGNN(nn.Module):
    def __init__(self, max_unit_id, num_attributes=8, hidden_dim=128):
        super(BattleGNN, self).__init__()
        
        # 调整各部分的维度分配，确保总和等于hidden_dim
        self.unit_embedding = nn.Embedding(max_unit_id + 1, hidden_dim//4)  # 32
        self.count_proj = nn.Linear(1, hidden_dim//4)  # 32
        self.side_embedding = nn.Embedding(2, hidden_dim//8)  # 16
        self.attr_proj = nn.Linear(num_attributes, hidden_dim//2)  # 64
        # 32+32+16+64=144，所以需要调整hidden_dim或各部分比例
        
        # 更好的方案是使用相同的hidden_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(3 + num_attributes, hidden_dim),  # unit_id, count, side + attributes
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 边类型编码
        self.edge_embedding = nn.Linear(1, hidden_dim)  # 边类型映射到相同维度
        
        # 图神经网络层
        self.conv1 = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=hidden_dim
        )
        
        self.conv2 = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=hidden_dim
        )
        
        # 预测层
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        
        # 节点特征编码
        x = self.node_encoder(x)
        
        # 边特征编码
        edge_emb = self.edge_embedding(edge_type.float().unsqueeze(-1))
        
        # 消息传递
        x = F.leaky_relu(self.conv1(x, edge_index, edge_emb))
        x = F.leaky_relu(self.conv2(x, edge_index, edge_emb))
        
        # 全局池化
        x = global_mean_pool(x, data.batch)
        
        return self.mlp(x)

if __name__ == "__main__":
    # 加载数据
    results_folder = "results"
    data_list = load_data_from_results_folder(results_folder)
    
    # 创建数据集
    dataset = BattleDataset(data_list, monster_csv_path=r'data\monster.csv')
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 初始化模型 - 使用max_unit_id而不是unit_types数量
    model = BattleGNN(max_unit_id=dataset.max_unit_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # 训练循环
    for epoch in range(50):
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")