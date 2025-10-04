# -*- coding: utf-8 -*-
"""
Step 7: 优化器和学习率调度器
设置优化器和学习率调度器，并可视化学习率曲线
"""

import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from transformers import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt

# 设置GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class CustomizedSchedule(_LRScheduler):
    """
    Noam / Transformer LR:
      lr = d_model**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))
    """

    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = float(d_model)
        self.warmup_steps = float(warmup_steps)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)  # 确保从 1 开始
        scale = self.d_model ** -0.5
        arg1 = step ** -0.5
        arg2 = step * (self.warmup_steps ** -1.5)
        lr = scale * min(arg1, arg2)
        return [lr for _ in self.base_lrs]

def plot_customized_lr_curve(optimizer, scheduler, total_steps: int, label: str = None):
    """
    绘制学习率曲线（支持传入已有 optimizer 和 scheduler）

    Args:
        optimizer (torch.optim.Optimizer): 优化器
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器
        total_steps (int): 总训练步数
        label (str): 图例标签，默认使用 scheduler 配置
    """
    lrs = []
    for step in range(total_steps):
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        lrs.append(lr)

    # 绘制曲线
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, total_steps + 1), lrs, label=label or "LR Curve")
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid(True)
    plt.show()

def create_optimizer_and_scheduler(model, learning_rate=5e-4, betas=(0.9, 0.98), 
                                 eps=1e-8, weight_decay=2e-5, epochs=600, 
                                 train_loader_length=1000, d_model=128):
    """
    创建优化器和学习率调度器
    
    Args:
        model: 模型
        learning_rate: 学习率
        betas: Adam优化器的beta参数
        eps: Adam优化器的epsilon参数
        weight_decay: 权重衰减
        epochs: 训练轮数
        train_loader_length: 训练集批次数
        d_model: 模型维度
    
    Returns:
        optimizer, scheduler
    """
    # 计算总训练步数
    num_training_steps = train_loader_length * epochs
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay
    )
    
    # 计算warmup步数
    warmup_steps = int(0.1 * num_training_steps)  # 10% 步数用作 warmup
    
    # 创建学习率调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=0.5,
    )
    
    return optimizer, scheduler, num_training_steps, warmup_steps

def analyze_optimizer_and_scheduler(optimizer, scheduler, num_training_steps):
    """
    分析优化器和调度器的配置
    """
    print(f"📊 优化器配置:")
    print(f"   优化器类型: {type(optimizer).__name__}")
    print(f"   学习率: {optimizer.param_groups[0]['lr']}")
    print(f"   Beta参数: {optimizer.param_groups[0]['betas']}")
    print(f"   Epsilon: {optimizer.param_groups[0]['eps']}")
    print(f"   权重衰减: {optimizer.param_groups[0]['weight_decay']}")
    
    print(f"\n📊 学习率调度器配置:")
    print(f"   调度器类型: {type(scheduler).__name__}")
    print(f"   总训练步数: {num_training_steps}")
    print(f"   Warmup步数: {scheduler.num_warmup_steps}")
    print(f"   训练步数: {scheduler.num_training_steps}")
    print(f"   周期数: {scheduler.num_cycles}")
    
    # 分析学习率变化
    print(f"\n📈 学习率变化分析:")
    print(f"   初始学习率: {scheduler.get_last_lr()[0]:.6f}")
    
    # 计算warmup结束时的学习率
    warmup_end_lr = scheduler.get_last_lr()[0]
    for _ in range(scheduler.num_warmup_steps):
        scheduler.step()
    warmup_end_lr = scheduler.get_last_lr()[0]
    print(f"   Warmup结束学习率: {warmup_end_lr:.6f}")
    
    # 计算最终学习率
    for _ in range(scheduler.num_training_steps - scheduler.num_warmup_steps):
        scheduler.step()
    final_lr = scheduler.get_last_lr()[0]
    print(f"   最终学习率: {final_lr:.6f}")

if __name__ == "__main__":
    print("=" * 60)
    print("Step 7: 优化器和学习率调度器")
    print("=" * 60)
    
    # 训练参数
    learning_rate = 5e-4
    betas = (0.9, 0.98)
    eps = 1e-8
    weight_decay = 2e-5
    epochs = 600
    train_loader_length = 1000  # 假设训练集有1000个batch
    d_model = 128
    
    print(f"🔧 训练参数:")
    print(f"   学习率: {learning_rate}")
    print(f"   Beta参数: {betas}")
    print(f"   Epsilon: {eps}")
    print(f"   权重衰减: {weight_decay}")
    print(f"   训练轮数: {epochs}")
    print(f"   训练批次数: {train_loader_length}")
    print(f"   模型维度: {d_model}")
    
    try:
        # 1. 创建示例模型（用于测试优化器）
        print(f"\n🔨 创建示例模型...")
        model = torch.nn.Linear(100, 10)  # 简单的线性模型用于测试
        
        # 2. 创建优化器和调度器
        print(f"\n🔨 创建优化器和调度器...")
        optimizer, scheduler, num_training_steps, warmup_steps = create_optimizer_and_scheduler(
            model=model,
            learning_rate=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            epochs=epochs,
            train_loader_length=train_loader_length,
            d_model=d_model
        )
        
        print(f"✅ 优化器和调度器创建完成！")
        
        # 3. 分析配置
        print(f"\n📊 分析优化器和调度器配置...")
        analyze_optimizer_and_scheduler(optimizer, scheduler, num_training_steps)
        
        # 4. 可视化学习率曲线
        print(f"\n📈 可视化学习率曲线...")
        
        # 重新创建调度器用于可视化（因为上面的已经被step了）
        scheduler_viz = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=0.5,
        )
        
        plot_customized_lr_curve(
            optimizer, 
            scheduler_viz, 
            total_steps=num_training_steps,
            label=f"d_model={d_model}, warmup={warmup_steps}"
        )
        
        print(f"\n✅ 优化器和学习率调度器设置完成！")
        
        # 5. 展示不同调度器的对比
        print(f"\n🔄 对比不同学习率调度器...")
        
        # 创建不同的调度器进行对比
        optimizers = []
        schedulers = []
        labels = []
        
        # AdamW + Cosine with Warmup
        opt1 = optim.AdamW(model.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)
        sched1 = get_cosine_schedule_with_warmup(opt1, warmup_steps, num_training_steps, num_cycles=0.5)
        optimizers.append(opt1)
        schedulers.append(sched1)
        labels.append("Cosine with Warmup")
        
        # AdamW + Linear with Warmup
        opt2 = optim.AdamW(model.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)
        sched2 = get_cosine_schedule_with_warmup(opt2, warmup_steps, num_training_steps, num_cycles=1.0)
        optimizers.append(opt2)
        schedulers.append(sched2)
        labels.append("Cosine (1 cycle)")
        
        # 绘制对比图
        plt.figure(figsize=(12, 6))
        for i, (opt, sched, label) in enumerate(zip(optimizers, schedulers, labels)):
            lrs = []
            for step in range(min(1000, num_training_steps)):  # 只显示前1000步
                sched.step()
                lr = sched.get_last_lr()[0]
                lrs.append(lr)
            
            plt.plot(range(1, len(lrs) + 1), lrs, label=label, linewidth=2)
        
        plt.ylabel("Learning Rate")
        plt.xlabel("Train Step")
        plt.title("Learning Rate Schedule Comparison")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print(f"\n✅ 学习率调度器对比完成！")
        
    except Exception as e:
        print(f"❌ 优化器和调度器设置失败: {e}")
        print("💡 请检查参数设置是否正确")
