import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def demonstrate_attention_scaling():
    """
    演示注意力机制中缩放的重要性
    """
    print("=== 注意力缩放演示 ===\n")
    
    # 模拟参数
    batch_size = 1
    seq_len = 10
    dk = 64  # 键的维度
    
    # 生成随机的Q和K
    torch.manual_seed(42)
    Q = torch.randn(batch_size, seq_len, dk)
    K = torch.randn(batch_size, seq_len, dk)
    
    print(f"Q形状: {Q.shape}")
    print(f"K形状: {K.shape}")
    print(f"dk = {dk}")
    print(f"√dk = {dk**0.5:.2f}\n")
    
    # 计算QK^T
    QK_T = torch.matmul(Q, K.transpose(-2, -1))
    print(f"QK^T形状: {QK_T.shape}")
    print(f"QK^T统计信息:")
    print(f"  均值: {QK_T.mean():.4f}")
    print(f"  标准差: {QK_T.std():.4f}")
    print(f"  最大值: {QK_T.max():.4f}")
    print(f"  最小值: {QK_T.min():.4f}\n")
    
    # 缩放后的QK^T
    QK_T_scaled = QK_T / (dk ** 0.5)
    print(f"QK^T/√dk统计信息:")
    print(f"  均值: {QK_T_scaled.mean():.4f}")
    print(f"  标准差: {QK_T_scaled.std():.4f}")
    print(f"  最大值: {QK_T_scaled.max():.4f}")
    print(f"  最小值: {QK_T_scaled.min():.4f}\n")
    
    # 计算softmax（取第一行作为示例）
    attention_weights_raw = F.softmax(QK_T[0, 0], dim=-1)
    attention_weights_scaled = F.softmax(QK_T_scaled[0, 0], dim=-1)
    
    print("注意力权重对比（第一行）:")
    print(f"未缩放: {attention_weights_raw}")
    print(f"缩放后: {attention_weights_scaled}")
    print(f"未缩放权重熵: {-torch.sum(attention_weights_raw * torch.log(attention_weights_raw + 1e-8)):.4f}")
    print(f"缩放后权重熵: {-torch.sum(attention_weights_scaled * torch.log(attention_weights_scaled + 1e-8)):.4f}")
    
    # 可视化对比
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(QK_T.flatten().numpy(), bins=50, alpha=0.7, label='QK^T')
    plt.hist(QK_T_scaled.flatten().numpy(), bins=50, alpha=0.7, label='QK^T/√dk')
    plt.xlabel('值')
    plt.ylabel('频次')
    plt.title('QK^T分布对比')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(attention_weights_raw.numpy(), 'o-', label='未缩放')
    plt.plot(attention_weights_scaled.numpy(), 's-', label='缩放后')
    plt.xlabel('位置')
    plt.ylabel('注意力权重')
    plt.title('注意力权重对比')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    # 计算所有行的注意力权重熵
    entropies_raw = []
    entropies_scaled = []
    
    for i in range(seq_len):
        weights_raw = F.softmax(QK_T[0, i], dim=-1)
        weights_scaled = F.softmax(QK_T_scaled[0, i], dim=-1)
        
        entropy_raw = -torch.sum(weights_raw * torch.log(weights_raw + 1e-8))
        entropy_scaled = -torch.sum(weights_scaled * torch.log(weights_scaled + 1e-8))
        
        entropies_raw.append(entropy_raw.item())
        entropies_scaled.append(entropy_scaled.item())
    
    plt.plot(entropies_raw, 'o-', label='未缩放')
    plt.plot(entropies_scaled, 's-', label='缩放后')
    plt.xlabel('序列位置')
    plt.ylabel('注意力熵')
    plt.title('注意力熵对比')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('attention_scaling_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n图表已保存为 'attention_scaling_demo.png'")

def explain_theory():
    """
    解释缩放的理论基础
    """
    print("\n=== 理论解释 ===\n")
    
    print("1. 数学原理:")
    print("   - 假设Q和K的元素都是独立的标准正态分布 N(0,1)")
    print("   - QK^T的每个元素是dk个独立随机变量的和")
    print("   - 根据中心极限定理，QK^T ~ N(0, dk)")
    print("   - 因此QK^T的方差约为dk")
    print("   - 除以√dk后，方差变为1，分布更稳定\n")
    
    print("2. 为什么需要缩放:")
    print("   - softmax函数对输入值很敏感")
    print("   - 当输入值过大时，exp(x)会溢出")
    print("   - 当输入值过小时，梯度会消失")
    print("   - 缩放后，输入值在合理范围内\n")
    
    print("3. 缩放的效果:")
    print("   - 防止数值溢出")
    print("   - 避免梯度消失")
    print("   - 使注意力分布更平滑")
    print("   - 提高训练稳定性\n")
    
    print("4. 实际应用:")
    print("   - 在Transformer中，dk通常是64或128")
    print("   - √dk约为8或11.3")
    print("   - 这个缩放因子是经验性的，但理论上有很好的解释")

if __name__ == "__main__":
    demonstrate_attention_scaling()
    explain_theory() 