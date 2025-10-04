# Transformer 训练分步脚本

这个文件夹包含了将原始 `train.py` 按照程序运行先后顺序分解的9个独立Python脚本，模拟用Jupyter运行代码查看结果的过程。

## 脚本说明

### Step 1: 环境检查 (`step1_env_check.py`)
- 检查 PyTorch 环境信息、GPU 状态
- 显示常用依赖库版本
- 设置GPU设备并启用优化选项

### Step 2: 数据加载 (`step2_data_loading.py`)
- 加载葡萄牙语-英语翻译数据集 (TED Talks)
- 显示数据集信息和样本预览
- 验证数据加载是否成功

### Step 3: Tokenizer训练 (`step3_tokenizer.py`)
- 训练葡萄牙语和英语的 ByteLevel BPE Tokenizer
- 测试tokenizer的编码/解码功能
- 保存tokenizer到本地文件

### Step 4: DataLoader构建 (`step4_dataloader.py`)
- 构建训练和验证 DataLoader
- 实现动态padding和序列过滤
- 测试DataLoader的输出格式

### Step 5: 位置编码可视化 (`step5_position_encoding.py`)
- 生成位置编码矩阵
- 可视化位置编码模式
- 分析位置编码的特性

### Step 6: Transformer模型构建 (`step6_model_building.py`)
- 实现完整的Transformer架构
- 包括Encoder、Decoder、MultiHeadAttention等组件
- 测试模型前向传播和参数统计

### Step 7: 优化器和学习率调度器 (`step7_optimizer_scheduler.py`)
- 设置AdamW优化器
- 配置Cosine学习率调度器with warmup
- 可视化学习率变化曲线

### Step 8: 模型训练过程 (`step8_training.py`)
- 实现完整的训练循环
- 包括损失计算、反向传播、梯度裁剪
- 训练过程监控和checkpoint保存
- 绘制训练曲线

### Step 9: 模型评估和翻译测试 (`step9_evaluation.py`)
- 实现模型推理和翻译功能
- 注意力权重可视化
- 翻译质量评估和性能分析

## 使用方法

### 顺序运行（推荐）
```bash
# 按顺序运行所有脚本
python step1_env_check.py
python step2_data_loading.py
python step3_tokenizer.py
python step4_dataloader.py
python step5_position_encoding.py
python step6_model_building.py
python step7_optimizer_scheduler.py
python step8_training.py
python step9_evaluation.py
```

### 单独运行
每个脚本都可以独立运行，但需要注意依赖关系：
- Step 1: 独立运行
- Step 2: 独立运行
- Step 3: 依赖 Step 2
- Step 4: 依赖 Step 2, 3
- Step 5: 独立运行
- Step 6: 独立运行
- Step 7: 独立运行
- Step 8: 依赖 Step 6, 7
- Step 9: 依赖 Step 6

## 注意事项

1. **数据路径**: 请确保数据文件路径正确，默认路径为：
   - 训练集: `/data2/workspace/yszhang/train_transformers/tensorflow_datasets/ted_pt_en_train.csv`
   - 验证集: `/data2/workspace/yszhang/train_transformers/tensorflow_datasets/ted_pt_en_test.csv`

2. **GPU设置**: 以下脚本设置了 `CUDA_VISIBLE_DEVICES="2"`，请根据实际情况修改：
   - step1_env_check.py (环境检查)
   - step5_position_encoding.py (位置编码)
   - step6_model_building.py (模型构建)
   - step7_optimizer_scheduler.py (优化器)
   - step8_training.py (模型训练)
   - step9_evaluation.py (模型评估)

3. **依赖库**: 确保安装了以下依赖：
   ```
   torch
   transformers
   datasets
   tokenizers
   matplotlib
   loguru
   ```

4. **内存要求**: 训练过程需要较大的内存，建议至少8GB RAM

5. **训练时间**: 完整训练需要较长时间，建议先用少量epoch测试

## 输出说明

每个脚本都会输出：
- 🔧 参数配置信息
- 📊 处理结果统计
- ✅ 成功完成提示
- ❌ 错误信息和解决建议

## 自定义修改

可以根据需要修改以下参数：
- 模型结构参数（层数、维度、注意力头数等）
- 训练超参数（学习率、批大小、训练轮数等）
- 数据路径和文件名
- GPU设备设置

## 故障排除

1. **数据加载失败**: 检查数据文件路径和格式
2. **GPU内存不足**: 减小批大小或模型维度
3. **依赖库缺失**: 安装所需的Python包
4. **权限问题**: 确保有读写checkpoint目录的权限
