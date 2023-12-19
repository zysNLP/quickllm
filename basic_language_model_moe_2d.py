# -*- coding: utf-8 -*- 
"""
    @Project ：quickllm 
    @File    ：basic_language_model_moe_2d.py
    @Author  ：ys
    @Time    ：2023/12/19 18:10 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 创建一些随机数据（替换为真实数据）
num_samples = 1000
num_features = 300  # 假设文本已经转换为固定大小的向量
num_classes = 10  # 假设有10个类别

# 随机生成数据和标签
X = np.random.randn(num_samples, num_features)
y = np.random.randint(0, num_classes, num_samples)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 定义 Dataset
class TextDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.long)


# 创建 DataLoader
train_dataset = TextDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TextDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


###模型定义
class TopKGating(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2):
        super(TopKGating, self).__init__()
        # 初始化线性层作为门控机制
        self.gate = nn.Linear(input_dim, num_experts)
        # 设置要选择的顶部专家数量
        self.top_k = top_k

    def forward(self, x):
        # 计算每个专家的分数
        gating_scores = self.gate(x)
        # 选取分数最高的 top_k 个专家，并返回它们的索引和 softmax 权重
        top_k_values, top_k_indices = torch.topk(F.softmax(gating_scores, dim=1), self.top_k)
        return top_k_indices, top_k_values


class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Expert, self).__init__()
        # 为每个专家定义一个简单的神经网络
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # 通过专家网络传递输入数据
        return self.net(x)


class MoE(nn.Module):
    def __init__(self, input_dim, num_classes, num_experts, top_k=2):
        super(MoE, self).__init__()
        # 设置专家数量
        self.num_experts = num_experts
        # 设置类别数量
        self.num_classes = num_classes
        # 初始化 TopK 门控层
        self.gating = TopKGating(input_dim, num_experts, top_k)
        # 创建专家网络的列表，每个专家是一个 Expert 实例
        self.experts = nn.ModuleList([Expert(input_dim, num_classes) for _ in range(num_experts)])

    def forward(self, x):
        # 获取批量大小
        batch_size = x.size(0)

        # 通过门控层获得 top_k 专家的索引和门控权重
        indices, gates = self.gating(x)  # 形状 indices：[batch_size, top_k], gates：[batch_size, top_k]

        # 准备收集选定专家的输出
        expert_outputs = torch.zeros(batch_size, indices.size(1), self.num_classes).to(x.device)

        # 遍历每个样本和其对应的 top_k 专家
        for i in range(batch_size):
            for j in range(indices.size(1)):
                expert_idx = indices[i, j].item()  # 获取专家的索引
                expert_outputs[i, j, :] = self.experts[expert_idx](x[i].unsqueeze(0))

        # 将门控权重扩展到与专家输出相同的维度
        gates = gates.unsqueeze(-1).expand(-1, -1, self.num_classes)  # 形状：[batch_size, top_k, num_classes]

        # 计算加权的专家输出的和
        output = (gates * expert_outputs).sum(1)
        return output, gates.sum(0)  # 返回模型输出和门控使用率以用于负载平衡损失计算


def moe_loss(output, target, gating_weights, lambda_balance=0.1):
    # 标准损失（例如交叉熵损失）
    # output 是模型的输出，target 是真实的标签
    standard_loss = F.cross_entropy(output, target)

    # 负载平衡损失
    # gating_weights 是门控权重，表示每个专家的使用率
    # 使用标准差来衡量各专家使用率的平衡程度
    balance_loss = torch.std(gating_weights)

    # 总损失
    # 结合标准损失和负载平衡损失，lambda_balance 是一个超参数，用于控制负载平衡损失在总损失中的比重
    total_loss = standard_loss + lambda_balance * balance_loss
    return total_loss


# 初始化模型
model = MoE(input_dim=num_features, num_classes=num_classes, num_experts=4, top_k=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs, gating_weights = model(features)
        loss = moe_loss(outputs, labels, gating_weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')


# def evaluate(model, data_loader):
#     model.eval()
#     predictions, true_labels = [], []
#     with torch.no_grad():
#         for features, labels in data_loader:
#             s = time.time()
#             outputs, _ = model(features)
#             e = time.time()
#             print(e - s)
#             predicted = torch.argmax(outputs, dim=1)
#             predictions.extend(predicted.tolist())
#             true_labels.extend(labels.tolist())
#     return accuracy_score(true_labels, predictions)
