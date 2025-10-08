# -*- coding: utf-8 -*-
"""
Step 5: RoPE位置编码可视化
生成和可视化RoPE位置编码矩阵
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import math


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


class RoPEPositionalEncoding:
    def __init__(self, max_len, d_model, nums_head=8, batch_size=1, device=None):
        self.max_len = max_len
        self.d_model = d_model
        self.nums_head = nums_head
        self.batch_size = batch_size

        if device is None:
            self.device = get_device()
        else:
            self.device = device

    def sinusoidal_position_embedding(self):
        """
        生成RoPE位置编码矩阵
        返回: [batch_size, nums_head, max_len, d_model]
        """
        # (max_len, 1)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(-1)

        # (d_model//2)
        ids = torch.arange(0, self.d_model // 2, dtype=torch.float)
        theta = torch.pow(10000, -2 * ids / self.d_model)

        # (max_len, d_model//2)
        embeddings = position * theta

        # (max_len, d_model//2, 2)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        # (batch_size, nums_head, max_len, d_model//2, 2)
        embeddings = embeddings.repeat((self.batch_size, self.nums_head, *([1] * len(embeddings.shape))))

        # (batch_size, nums_head, max_len, d_model)
        embeddings = torch.reshape(embeddings, (self.batch_size, self.nums_head, self.max_len, self.d_model))

        return embeddings.to(self.device)

    def get_rope_embedding_matrix(self):
        """
        获取RoPE位置编码矩阵用于可视化
        返回: [max_len, d_model]
        """
        # 只取第一个batch和第一个head的位置编码
        rope_emb = self.sinusoidal_position_embedding()
        return rope_emb[0, 0].detach().cpu()  # [max_len, d_model]

    def get_rotation_matrices(self, positions_to_show=5):
        """
        获取旋转矩阵的可视化数据
        对于每个位置，计算旋转矩阵对向量的影响
        """
        # 生成一些测试向量
        test_vectors = []
        for i in range(4):
            angle = i * math.pi / 4  # 0, 45, 90, 135度
            vec = torch.tensor([math.cos(angle), math.sin(angle)], dtype=torch.float32)
            test_vectors.append(vec)

        # 计算旋转效果
        rotation_data = []
        for pos in range(min(positions_to_show, self.max_len)):
            pos_emb = self.sinusoidal_position_embedding()[0, 0, pos]  # 取第一个位置编码

            rotated_vectors = []
            for vec in test_vectors:
                # 简化版的旋转计算（只考虑前两个维度）
                cos_theta = pos_emb[1]  # cos分量
                sin_theta = pos_emb[0]  # sin分量

                # 旋转矩阵 [cos, -sin; sin, cos]
                rotation_matrix = torch.tensor([
                    [cos_theta, -sin_theta],
                    [sin_theta, cos_theta]
                ])

                rotated_vec = torch.matmul(rotation_matrix, vec)
                rotated_vectors.append({
                    'original': vec.numpy(),
                    'rotated': rotated_vec.numpy(),
                    'position': pos
                })

            rotation_data.append(rotated_vectors)

        return rotation_data


def plot_rope_embedding(rope_embedding: torch.Tensor):
    """
    可视化RoPE位置编码矩阵
    参数:
        rope_embedding: [L, D] 的张量
    """
    pe = rope_embedding.numpy()  # [L, D]

    plt.figure(figsize=(15, 10))

    # 主图：RoPE位置编码的热力图
    plt.subplot(2, 2, 1)
    plt.pcolormesh(pe, cmap='RdBu')
    plt.xlabel("Depth (d_model)")
    plt.xlim((0, pe.shape[1]))
    plt.ylabel("Position (pos)")
    plt.colorbar(label='Encoding Value')
    plt.title("RoPE Positional Encoding Heatmap")

    # 子图：正弦和余弦分量
    plt.subplot(2, 2, 2)
    positions_to_plot = min(8, pe.shape[0])
    for pos in range(positions_to_plot):
        plt.plot(pe[pos, :], label=f'Pos {pos}', alpha=0.7, linewidth=1)
    plt.xlabel("Dimension")
    plt.ylabel("Encoding Value")
    plt.title("RoPE Encoding by Position")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 子图：正弦分量（偶数维度）
    plt.subplot(2, 2, 3)
    sin_components = pe[:, 0::2]  # 所有位置的sin分量
    for pos in range(positions_to_plot):
        plt.plot(sin_components[pos, :25], label=f'Pos {pos}', alpha=0.7, linewidth=2)
    plt.xlabel("Sin Component Index")
    plt.ylabel("Sin Value")
    plt.title("RoPE Sin Components (First 25)")
    plt.legend()

    # 子图：余弦分量（奇数维度）
    plt.subplot(2, 2, 4)
    cos_components = pe[:, 1::2]  # 所有位置的cos分量
    for pos in range(positions_to_plot):
        plt.plot(cos_components[pos, :25], label=f'Pos {pos}', alpha=0.7, linewidth=2)
    plt.xlabel("Cos Component Index")
    plt.ylabel("Cos Value")
    plt.title("RoPE Cos Components (First 25)")
    plt.legend()

    plt.tight_layout()
    plt.savefig('images/step5_rope_positional_encoding1.png', dpi=300, bbox_inches='tight')


def plot_rotation_effect(rotation_data):
    """
    可视化旋转效果
    """
    plt.figure(figsize=(12, 8))

    positions = len(rotation_data)
    vectors_per_pos = len(rotation_data[0]) if rotation_data else 0

    for pos_idx, pos_vectors in enumerate(rotation_data):
        plt.subplot(2, 3, pos_idx + 1)

        # 绘制单位圆
        circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.3)
        plt.gca().add_patch(circle)

        colors = ['red', 'blue', 'green', 'orange']

        for vec_idx, vec_data in enumerate(pos_vectors):
            orig = vec_data['original']
            rotated = vec_data['rotated']
            color = colors[vec_idx % len(colors)]

            # 绘制原始向量
            plt.arrow(0, 0, orig[0], orig[1],
                      head_width=0.05, head_length=0.1,
                      fc=color, ec=color, alpha=0.6,
                      label=f'Vec{vec_idx}' if pos_idx == 0 else "")

            # 绘制旋转后的向量
            plt.arrow(0, 0, rotated[0], rotated[1],
                      head_width=0.05, head_length=0.1,
                      fc=color, ec=color, alpha=1.0, linestyle='-')

        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.grid(True, alpha=0.3)
        plt.gca().set_aspect('equal')
        plt.title(f'Position {pos_idx}')
        plt.xlabel('X')
        plt.ylabel('Y')

    plt.suptitle('RoPE Rotation Effect on Different Positions')
    plt.tight_layout()
    plt.savefig('images/step5_rope_positional_encoding2.png', dpi=300, bbox_inches='tight')


def analyze_rope_encoding(rope_embedding: torch.Tensor):
    """
    分析RoPE位置编码的特性
    """
    pe = rope_embedding.numpy()  # [L, D]

    print(f"📊 RoPE位置编码分析:")
    print(f"   序列长度: {pe.shape[0]}")
    print(f"   嵌入维度: {pe.shape[1]}")
    print(f"   数值范围: [{pe.min():.4f}, {pe.max():.4f}]")
    print(f"   均值: {pe.mean():.4f}")
    print(f"   标准差: {pe.std():.4f}")

    # 分析正弦和余弦分量
    sin_components = pe[:, 0::2]
    cos_components = pe[:, 1::2]

    print(f"\n🔍 分量分析:")
    print(f"   正弦分量范围: [{sin_components.min():.4f}, {sin_components.max():.4f}]")
    print(f"   余弦分量范围: [{cos_components.min():.4f}, {cos_components.max():.4f}]")
    print(f"   正弦分量均值: {sin_components.mean():.4f}")
    print(f"   余弦分量均值: {cos_components.mean():.4f}")

    # 分析位置间的相似性
    print(f"\n📈 位置相似性分析:")
    positions_to_check = min(5, pe.shape[0])
    for i in range(positions_to_check):
        for j in range(i + 1, positions_to_check):
            similarity = np.corrcoef(pe[i], pe[j])[0, 1]
            print(f"   位置 {i} 与位置 {j} 的相似度: {similarity:.4f}")


def verify_rope_encoding(rope_embedding: torch.Tensor):
    """
    验证RoPE位置编码的正确性
    """
    pe = rope_embedding.numpy()  # [L, D]

    print(f"\n✅ RoPE位置编码验证:")

    # 检查数值范围
    assert pe.min() >= -1.0 and pe.max() <= 1.0, "RoPE编码值超出预期范围"
    print("   ✓ 数值范围验证通过")

    # 检查正弦分量
    sin_components = pe[:, 0::2]
    assert abs(sin_components[0, 0]) < 1e-6, f"位置0的正弦值应该接近0，实际为{sin_components[0, 0]}"
    print("   ✓ 正弦分量验证通过")

    # 检查余弦分量
    cos_components = pe[:, 1::2]
    assert abs(cos_components[0, 0] - 1.0) < 1e-6, f"位置0的余弦值应该接近1，实际为{cos_components[0, 0]}"
    print("   ✓ 余弦分量验证通过")

    # 检查不同位置编码是否不同
    unique_positions = len(set(tuple(row) for row in pe))
    assert unique_positions == pe.shape[0], "存在重复的位置编码"
    print("   ✓ 位置唯一性验证通过")

    print("   🎉 所有验证通过！RoPE位置编码生成正确。")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 5: RoPE位置编码可视化")
    print("=" * 60)

    # 参数设置
    max_length = 50  # 最大序列长度
    d_model = 128  # 嵌入维度
    nums_head = 8  # 注意力头数

    print(f"🔧 RoPE位置编码参数:")
    print(f"   最大序列长度: {max_length}")
    print(f"   嵌入维度: {d_model}")
    print(f"   注意力头数: {nums_head}")

    try:
        # 获取设备
        device = get_device()

        # 1. 初始化RoPE可视化器
        print(f"\n🔨 初始化RoPE可视化器...")
        rope_viz = RoPEPositionalEncoding(
            max_len=max_length,
            d_model=d_model,
            nums_head=nums_head,
            batch_size=1,
            device=device
        )

        # 2. 生成RoPE位置编码
        print(f"🔨 生成RoPE位置编码...")
        rope_embedding = rope_viz.get_rope_embedding_matrix()

        print(f"✅ RoPE位置编码生成完成！")
        print(f"📊 位置编码形状: {rope_embedding.shape}")

        # 3. 验证RoPE位置编码
        verify_rope_encoding(rope_embedding)

        # 4. 分析RoPE位置编码
        print(f"\n🔍 分析RoPE位置编码特性...")
        analyze_rope_encoding(rope_embedding)

        # 5. 可视化RoPE位置编码矩阵
        print(f"\n📊 可视化RoPE位置编码矩阵...")
        plot_rope_embedding(rope_embedding)

        # 6. 可视化旋转效果
        print(f"\n🔄 可视化旋转效果...")
        rotation_data = rope_viz.get_rotation_matrices(positions_to_show=6)
        plot_rotation_effect(rotation_data)

        print(f"\n✅ RoPE位置编码可视化完成！")

        # 7. 展示RoPE位置编码的数值
        print(f"\n📝 RoPE位置编码数值示例 (前5个位置，前10个维度):")
        pe_np = rope_embedding.numpy()
        for i in range(min(5, pe_np.shape[0])):
            values_str = " ".join([f"{x:6.3f}" for x in pe_np[i, :10]])
            print(f"   位置 {i:2d}: [{values_str}]")

    except Exception as e:
        print(f"❌ RoPE位置编码生成失败: {e}")
        import traceback

        traceback.print_exc()
        print("💡 建议检查:")
        print("   1. PyTorch 是否正确安装")
        print("   2. 设备是否可用")
        print("   3. 参数设置是否合理")