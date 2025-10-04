# -*- coding: utf-8 -*-
"""
Step 1: 环境检查和依赖库版本
检查 PyTorch 环境信息、GPU 状态，以及常用依赖库版本
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
from loguru import logger

# 设置GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def check_env():
    """
    检查 PyTorch 环境信息、GPU 状态，以及常用依赖库版本。
    返回推荐的 device ('cuda' 或 'cpu')。
    """
    logger.info("===== PyTorch & 系统信息 =====")
    logger.info(f"torch.__version__: {torch.__version__}")
    logger.info(f"python version: {sys.version_info}")

    logger.info("\n===== 常用库版本 =====")
    for module in (mpl, np, pd, torch):
        logger.info(f"{module.__name__}: {module.__version__}")

    logger.info("\n===== GPU 检查 =====")
    logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    logger.info(f"torch.version.cuda: {torch.version.cuda}")
    try:
        logger.info(f"cudnn version: {torch.backends.cudnn.version()}")
    except Exception as e:
        logger.info(f"cudnn version: N/A, {e}")

    if torch.cuda.is_available():
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"Current device id: {torch.cuda.current_device()}")
        logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
        logger.info(f"bfloat16 supported: {torch.cuda.is_bf16_supported()}")

        # 启用 TF32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        device = "cuda"
    else:
        logger.info("⚠️ 没检测到 CUDA，可强制 device='cpu' 运行，但速度会慢")
        device = "cpu"

    logger.info(f"\n推荐使用 device: {device}")
    return device

if __name__ == "__main__":
    print("=" * 60)
    print("Step 1: 环境检查和依赖库版本")
    print("=" * 60)
    
    # 检查环境
    device = check_env()
    
    print(f"\n✅ 环境检查完成！")
    print(f"📱 推荐使用设备: {device}")
    print(f"🔧 PyTorch版本: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"🚀 GPU可用: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  GPU不可用，将使用CPU")
