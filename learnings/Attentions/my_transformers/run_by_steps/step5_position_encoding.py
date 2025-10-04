# -*- coding: utf-8 -*-
"""
Step 5: 位置编码可视化
生成和可视化位置编码矩阵
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np


def get_device():
    """自动检测可用设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ 使用 GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon GPU
        print("✅ 使用 Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("✅ 使用 CPU")
    return device


def get_position_embedding(sentence_length: int, d_model: int, device=None, dtype=torch.float32):
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
        device = get_device()

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

    plt.figure(figsize=(12, 8))

    # 主图：位置编码的热力图
    plt.subplot(2, 1, 1)
    plt.pcolormesh(pe, cmap='RdBu')  # L × D 矩阵
    plt.xlabel("Depth (d_model)")
    plt.xlim((0, pe.shape[1]))
    plt.ylabel("Position (pos)")
    plt.colorbar(label='Encoding Value')
    plt.title("Positional Encoding Visualization")

    # 子图：前几个位置的编码模式
    plt.subplot(2, 1, 2)
    positions_to_plot = min(8, pe.shape[0])
    for pos in range(positions_to_plot):
        plt.plot(pe[pos, :50], label=f'Pos {pos}', alpha=0.7, linewidth=2)
    plt.xlabel("Dimension")
    plt.ylabel("Encoding Value")
    plt.title("Positional Encoding Patterns (First 50 Dimensions)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    #plt.show()
    plt.savefig('step5_positional_encoding.png', dpi=300, bbox_inches='tight')


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
    positions_to_check = min(5, pe.shape[0])
    for i in range(positions_to_check):
        for j in range(i + 1, positions_to_check):
            similarity = np.corrcoef(pe[i], pe[j])[0, 1]
            print(f"   位置 {i} 与位置 {j} 的相似度: {similarity:.4f}")

    # 分析维度间的相关性
    print(f"\n📈 维度分析:")
    even_dims = pe[:, 0::2]  # 偶数维度 (sin)
    odd_dims = pe[:, 1::2]  # 奇数维度 (cos)
    print(f"   偶数维度(sin)范围: [{even_dims.min():.4f}, {even_dims.max():.4f}]")
    print(f"   奇数维度(cos)范围: [{odd_dims.min():.4f}, {odd_dims.max():.4f}]")


def verify_position_encoding(position_embedding: torch.Tensor):
    """
    验证位置编码的正确性
    """
    pe = position_embedding.detach().cpu().numpy()[0]  # [L, D]

    print(f"\n✅ 位置编码验证:")

    # 检查数值范围
    assert pe.min() >= -1.0 and pe.max() <= 1.0, "位置编码值超出预期范围"
    print("   ✓ 数值范围验证通过")

    # 检查交替模式
    even_dims = pe[:, 0::2]  # 应该是sin
    odd_dims = pe[:, 1::2]  # 应该是cos

    # 对于第一个位置，检查sin(0)应该接近0
    first_pos_sin = even_dims[0, 0]
    assert abs(first_pos_sin) < 1e-6, f"第一个位置的sin值应该接近0，实际为{first_pos_sin}"
    print("   ✓ 正弦余弦交替模式验证通过")

    # 检查不同位置编码是否不同
    unique_positions = len(set(tuple(row) for row in pe))
    assert unique_positions == pe.shape[0], "存在重复的位置编码"
    print("   ✓ 位置唯一性验证通过")

    print("   🎉 所有验证通过！位置编码生成正确。")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 5: 位置编码可视化")
    print("=" * 60)

    # 参数设置
    max_length = 30  # 最大序列长度
    d_model = 128  # 嵌入维度

    print(f"🔧 位置编码参数:")
    print(f"   最大序列长度: {max_length}")
    print(f"   嵌入维度: {d_model}")

    try:
        # 获取设备
        device = get_device()

        # 1. 生成位置编码
        print(f"\n🔨 生成位置编码...")
        position_embedding = get_position_embedding(max_length, d_model, device=device)

        print(f"✅ 位置编码生成完成！")
        print(f"📊 位置编码形状: {position_embedding.shape}")
        print(f"💻 使用设备: {device}")

        # 2. 验证位置编码
        verify_position_encoding(position_embedding)

        # 3. 分析位置编码
        print(f"\n🔍 分析位置编码特性...")
        analyze_position_encoding(position_embedding)

        # 4. 可视化位置编码
        print(f"\n📊 可视化位置编码...")
        plot_position_embedding(position_embedding)

        print(f"\n✅ 位置编码可视化完成！")

        # 5. 展示位置编码的数值
        print(f"\n📝 位置编码数值示例 (前5个位置，前10个维度):")
        pe_np = position_embedding.detach().cpu().numpy()[0]
        for i in range(min(5, pe_np.shape[0])):
            values_str = " ".join([f"{x:6.3f}" for x in pe_np[i, :10]])
            print(f"   位置 {i:2d}: [{values_str}]")


    except Exception as e:
        print(f"❌ 位置编码生成失败: {e}")
        import traceback

        traceback.print_exc()
        print("💡 建议检查:")
        print("   1. PyTorch 是否正确安装")
        print("   2. 设备是否可用")
        print("   3. 参数设置是否合理")