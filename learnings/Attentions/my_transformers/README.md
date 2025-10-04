# Transformer 训练项目

## 项目概述

本项目实现了完整的 Transformer 模型训练流程，用于葡萄牙语到英语的机器翻译任务。项目包含从数据预处理、模型构建、训练到评估的完整流程。

## Transformer 训练过程总结

### 一句话总结
Transformer 训练过程包括环境检查、数据加载、Tokenizer构建、DataLoader准备、位置编码生成、模型构建、优化器配置、损失函数设置和模型训练评估等九个核心步骤。

### 详细过程描述

Transformer 的训练过程是一个系统性的端到端流程，主要包含以下九个关键步骤：

1. **环境检查**：首先检查 PyTorch 环境、GPU 状态和依赖库版本，确保训练环境配置正确
2. **数据加载**：加载葡萄牙语-英语翻译数据集（TED Talks），进行数据格式验证和样本预览
3. **Tokenizer构建**：训练葡萄牙语和英语的 ByteLevel BPE Tokenizer，实现文本的编码解码功能
4. **DataLoader准备**：构建训练和验证的 DataLoader，实现动态padding和序列长度过滤
5. **位置编码生成**：生成并可视化位置编码矩阵，为模型提供位置信息
6. **模型构建**：构建完整的 Transformer 架构，包括 Encoder、Decoder、MultiHeadAttention 等组件
7. **优化器配置**：设置 AdamW 优化器和 Cosine 学习率调度器，实现 warmup 机制
8. **损失函数设置**：配置交叉熵损失函数，支持 padding token 的忽略处理
9. **模型训练评估**：执行训练循环，包括前向传播、反向传播、梯度裁剪、checkpoint 保存和验证评估

整个训练过程采用 Teacher Forcing 策略，使用交叉熵损失进行监督学习，通过梯度裁剪防止梯度爆炸，并定期保存模型检查点以确保训练的可恢复性。训练完成后，模型可以进行推理和翻译任务，并支持注意力权重的可视化分析。

## 项目结构

```
learnings/Attentions/my_transformers/
├── train.py                    # 完整的训练脚本
├── run_by_steps/              # 分步训练脚本
│   ├── step1_env_check.py     # 环境检查
│   ├── step2_data_loading.py  # 数据加载
│   ├── step3_tokenizer.py     # Tokenizer构建
│   ├── step4_dataloader.py    # DataLoader准备
│   ├── step5_position_encoding.py # 位置编码
│   ├── step6_model_building.py   # 模型构建
│   ├── step7_optimizer_scheduler.py # 优化器配置
│   ├── step8_training.py      # 模型训练
│   ├── step9_evaluation.py    # 模型评估
│   └── README.md              # 使用说明
└── README.md                  # 项目说明
```

## 使用方法

### 完整训练
```bash
python train.py
```

### 分步训练
```bash
cd run_by_steps
python step1_env_check.py
python step2_data_loading.py
# ... 依次运行其他步骤
```

## 技术特点

- **完整的 Transformer 实现**：包含 Encoder、Decoder、MultiHeadAttention 等核心组件
- **端到端训练流程**：从数据预处理到模型评估的完整流程
- **模块化设计**：每个步骤独立可运行，便于学习和调试
- **GPU 加速支持**：支持 CUDA 加速训练
- **可视化功能**：包含位置编码、学习率曲线、注意力权重等可视化
- **Checkpoint 机制**：支持训练中断恢复和模型保存

## 依赖要求

- Python 3.7+
- PyTorch 1.8+
- transformers
- datasets
- tokenizers
- matplotlib
- loguru
