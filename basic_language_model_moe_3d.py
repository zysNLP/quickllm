# -*- coding: utf-8 -*- 
# @Time : 2023/12/13 02:09 
# @Author : ys 
# @File : basic_language_model_moe_3d.py

import torch
from torch import nn
from quickllm.layers.moe import MoE, TextDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


if __name__ == "__main__":

    # 构造一些数据
    num_samples = 1000
    num_features = 128  # 假设文本已经转换为固定大小的向量
    num_classes = 10    # 假设有10个类别
    hidden_size = 64
    num_experts = 16

    # 随机生成数据和标签
    X = np.random.randn(num_samples, num_features, hidden_size)
    y = np.random.randint(0, num_classes, num_samples)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TextDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    test_dataset = TextDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    moe = MoE(
        dim=hidden_size,  # 输入张量的维度
        num_experts=num_experts,  # 专家数量，可以增加该参数而不增加计算量
        hidden_dim=hidden_size * 4,  # 每个专家网络中的隐藏层维度，默认为 4 倍输入维度
        activation=nn.LeakyReLU,  # 使用的激活函数，默认为 GELU
        second_policy_train='random',  # 使用的第二名专家的训练策略
        second_policy_eval='random',  # 使用的第二名专家的验证策略
        second_threshold_train=0.2,  # 训练时使用的第二名专家阈值
        second_threshold_eval=0.2,  # 测试时使用的第二名专家阈值
        capacity_factor_train=1.25,  # 每个专家网络在单个批次中的固定容量，需要额外的容量以防门控不平衡
        capacity_factor_eval=2.,  # capacity_factor_* 应设置为 >=1 的值
        loss_coef=1e-2  # 辅助专家平衡辅助损失的乘数
    )
    # inputs = torch.randn(4, 1024, 512)
    # out, aux_loss = moe(inputs)  # (4, 1024, 512), (1,)
    # print(out.shape, aux_loss.shape)

    optimizer = torch.optim.Adam(moe.parameters(), lr=0.001)

    num_epochs = 100
    for epoch in range(num_epochs):
        moe.train()
        total_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs, loss = moe(features)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')