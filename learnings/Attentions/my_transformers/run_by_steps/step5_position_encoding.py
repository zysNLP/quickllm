# -*- coding: utf-8 -*-
"""
Step 5: 位置编码可视化
生成和可视化位置编码矩阵
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# 设置GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def get_position_embedding(sentence_length: int, d_model: int, device="cuda", dtype=torch.float32):
    """
    返回 position 对应的 embedding 矩阵
    形状: [1, sentence_length, d_model]
    """

    def get_angles(pos: torch.Tensor, i: torch.Tensor, d_model: int):
        """
        获取单词 pos 对应 embedding 的角度
        pos: [sentence_length, 1]
        i  : [1, d_model]
        return: [sentence_length, d_model]
        """
        angle_rates = 1.0 / torch.pow(
            10000,
            (2 * torch.div(i, 2, rounding_mode='floor')).float() / d_model
        )
        return pos.float() * angle_rates

    if device is None:
        device = torch.device("cpu")

    pos = torch.arange(sentence_length, device=device).unsqueeze(1)  # [L, 1]
    i = torch.arange(d_model, device=device).unsqueeze(0)  # [1, D]

    angle_rads = get_angles(pos, i, d_model)  # [L, D]

    # 偶数下标：sin
    sines = torch.sin(angle_rads[:, 0::2])
    # 奇数下标：cos
    cosines = torch.cos(angle_rads[:, 1::2])

    # 拼接还原成 [L, D]
    position_embedding = torch.zeros((sentence_length, d_model), device=device, dtype=dtype)
    position_embedding[:, 0::2] = sines
    position_embedding[:, 1::2] = cosines

    # 增加 batch 维度 [1, L, D]
    position_embedding = position_embedding.unsqueeze(0)

    return position_embedding

def plot_position_embedding(position_embedding: torch.Tensor):
    """
    可视化位置编码矩阵
    参数:
        position_embedding: [1, L, D] 的张量
    """
    # 转到 CPU，并转成 numpy
    pe = position_embedding.detach().cpu().numpy()[0]  # [L, D]

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(pe, cmap='RdBu')  # L × D 矩阵
    plt.xlabel("Depth (d_model)")
    plt.xlim((0, pe.shape[1]))
    plt.ylabel("Position (pos)")
    plt.colorbar()
    plt.title("Positional Encoding Visualization")
    plt.show()

def analyze_position_encoding(position_embedding: torch.Tensor):
    """
    分析位置编码的特性
    """
    pe = position_embedding.detach().cpu().numpy()[0]  # [L, D]
    
    print(f"📊 位置编码分析:")
    print(f"   序列长度: {pe.shape[0]}")
    print(f"   嵌入维度: {pe.shape[1]}")
    print(f"   数值范围: [{pe.min():.4f}, {pe.max():.4f}]")
    print(f"   均值: {pe.mean():.4f}")
    print(f"   标准差: {pe.std():.4f}")
    
    # 分析不同位置的相似性
    print(f"\n🔍 位置相似性分析:")
    for i in range(min(5, pe.shape[0])):
        for j in range(i+1, min(5, pe.shape[0])):
            similarity = np.corrcoef(pe[i], pe[j])[0, 1]
            print(f"   位置 {i} 与位置 {j} 的相似度: {similarity:.4f}")

if __name__ == "__main__":
    print("=" * 60)
    print("Step 5: 位置编码可视化")
    print("=" * 60)
    
    # 参数设置
    max_length = 30  # 最大序列长度
    d_model = 128   # 嵌入维度
    
    print(f"🔧 位置编码参数:")
    print(f"   最大序列长度: {max_length}")
    print(f"   嵌入维度: {d_model}")
    
    try:
        # 1. 生成位置编码
        print(f"\n🔨 生成位置编码...")
        position_embedding = get_position_embedding(max_length, d_model)
        
        print(f"✅ 位置编码生成完成！")
        print(f"📊 位置编码形状: {position_embedding.shape}")
        
        # 2. 分析位置编码
        print(f"\n🔍 分析位置编码特性...")
        analyze_position_encoding(position_embedding)
        
        # 3. 可视化位置编码
        print(f"\n📊 可视化位置编码...")
        plot_position_embedding(position_embedding)
        
        print(f"\n✅ 位置编码可视化完成！")
        
        # 4. 展示位置编码的数值
        print(f"\n📝 位置编码数值示例 (前5个位置，前10个维度):")
        pe_np = position_embedding.detach().cpu().numpy()[0]
        for i in range(min(5, pe_np.shape[0])):
            print(f"   位置 {i}: {pe_np[i, :10]}")
        
    except Exception as e:
        print(f"❌ 位置编码生成失败: {e}")
        print("💡 请检查参数设置是否正确")
