#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 测试多卡GPU设置
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,5,6,7"

import torch

print("=== GPU 测试 ===")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} 显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")

# 测试多卡模型
if torch.cuda.device_count() > 1:
    print("\n=== 多卡模型测试 ===")
    model = torch.nn.Linear(10, 1)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    print(f"模型已部署到 {torch.cuda.device_count()} 张GPU")
    
    # 测试前向传播
    x = torch.randn(32, 10).cuda()
    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print("✅ 多卡训练测试成功！")
else:
    print("❌ 只检测到单卡，多卡训练无法启用")
