# VLM GRPO (Vision-Language Model Group Relative Policy Optimization) 算法详解与实现指南

本文代码参见：[https://github.com/zysNLP/quickllm/tree/main/learnings/dim_grpo](https://github.com/zysNLP/quickllm/tree/main/learnings/dim_grpo)；感谢star。本文内容非常生动形象、但也非常长非常详细，请参照代码逐行耐心查看

## 1. VLM GRPO算法简介

视觉语言模型组相对策略优化（Vision-Language Model Group Relative Policy Optimization, VLM GRPO）是一种专门用于多模态任务的强化学习算法。与传统的GRPO不同，VLM GRPO能够同时处理**文本和图像输入**，通过组内相对比较来优化模型对产品尺寸的提取能力。

### 🎯 **为什么需要VLM GRPO？**

在产品尺寸提取任务中，传统的方法面临以下挑战：
1. **多模态理解**：需要同时理解产品描述文本和产品图片
2. **尺寸精度**：长、宽、高的提取需要高精度
3. **产品多样性**：不同产品的尺寸特征差异很大
4. **标注成本**：人工标注产品尺寸成本高昂

VLM GRPO通过**多模态组内相对比较**解决了这些问题：

```python
# 传统方法：单一模态处理
文本: "这是一个红色的杯子" → 尺寸: [10, 8, 15] (不准确)

# VLM GRPO：多模态联合处理
文本: "这是一个红色的杯子"
图片: [产品图片]
候选1: {"length": 12, "width": 8, "height": 15}  → 奖励: +0.9 (接近真实)
候选2: {"length": 10, "width": 8, "height": 12}  → 奖励: +0.7 (部分准确)
候选3: {"length": 20, "width": 15, "height": 25} → 奖励: +0.1 (偏差较大)
候选4: {"length": 0, "width": 0, "height": 0}    → 奖励: +0.0 (完全错误)
```

## 2. 核心概念：4+3理解法

### 2.1 四个模型

#### 2.1.1 策略模型（Actor Model）
- **作用**：待优化的主模型，负责从文本和图像中提取产品尺寸
- **参数更新**：✅ 参与训练，通过VLM GRPO损失进行优化
- **代码位置**：`model = AutoModel.from_pretrained(...)`

#### 2.1.2 参考模型（Reference Model）
- **作用**：防止策略模型偏离原始模型太远，提供KL散度约束
- **参数更新**：❌ 不参与训练，权重固定
- **代码位置**：`reference_model = AutoModel.from_pretrained(...)`

#### 2.1.3 图像处理器（Image Processor）
- **作用**：处理输入图像，转换为模型可理解的格式
- **参数更新**：❌ 不参与训练，权重固定
- **代码位置**：`image_processor = AutoImageProcessor.from_pretrained(...)`

#### 2.1.4 分词器（Tokenizer）
- **作用**：处理文本输入，转换为token序列
- **参数更新**：❌ 不参与训练，权重固定
- **代码位置**：`tokenizer = AutoTokenizer.from_pretrained(...)`

### 2.2 三个核心机制

#### 2.2.1 多模态输入处理（Multimodal Input Processing）
```python
def build_prompt(input_text, images):
    """构建多模态提示词"""
    prompt = (
        f"基于以下产品信息，包括文本和多张图片，提取产品的长、宽、高尺寸。\n"
        f"产品文本描述: {input_text}\n"
        "请严格按照以下JSON格式输出，不要包含任何额外说明或markdown标记：\n"
        "{\"length\": 数字, \"width\": 数字, \"height\": 数字, \"unit\": \"cm\"}"
    )
    return prompt

# 使用模型进行多模态推理
user_msgs = [{"role": "user", "content": prompt_str, "images": images}]
answer = model.chat(msgs=user_msgs, tokenizer=tokenizer, ...)
```

#### 2.2.2 尺寸排序奖励（Dimension Sorting Reward）
```python
def calculate_reward(answer, sample_label):
    """计算基于尺寸排序的相对奖励"""
    # 1. 解析并排序标签和预测的尺寸
    label_sorted = sorted([length, width, height], reverse=True)
    answer_sorted = sorted([pred_length, pred_width, pred_height], reverse=True)
    
    # 2. 计算排序后对应位置的相对误差
    for i in range(min_len):
        rel_error = abs(answer_sorted[i] - label_sorted[i]) / label_sorted[i]
        total_error += rel_error
    
    # 3. 奖励 = 1 - 平均相对误差
    reward = max(0.0, 1.0 - avg_error)
    return reward
```

#### 2.2.3 组内相对比较（Group Relative Comparison）
```python
# 对同一个产品生成多个候选尺寸
for i in range(num_rollouts):
    candidate_answer = model.generate(text, images)
    reward = calculate_reward(candidate_answer, ground_truth)
    rewards.append(reward)

# 计算相对优势
advantages = (rewards - rewards.mean()) / (rewards.std() + eps)
```

## 3. 数学推导过程

### 3.1 基础概念

#### 3.1.1 多模态轨迹
在VLM GRPO中：
- **轨迹**：一次完整的尺寸提取过程
- **状态**：当前的文本和图像输入
- **动作**：生成下一个尺寸预测token
- **奖励**：基于尺寸排序的相对误差

多模态轨迹定义：
$$
\tau = (s_0, a_0, s_1, a_1, \ldots, s_{T-1}, a_{T-1})
$$

其中$s_t$包含文本和图像信息。

#### 3.1.2 尺寸排序奖励函数
VLM GRPO的核心创新在于尺寸排序奖励机制：
$$
R(\tau) = \max(0, 1 - \frac{1}{N} \sum_{i=1}^{N} \frac{|d_i^{pred} - d_i^{true}|}{d_i^{true}})
$$

其中：
- $d_i^{pred}$：预测尺寸的排序值
- $d_i^{true}$：真实尺寸的排序值
- $N$：尺寸数量（通常为3：长、宽、高）

### 3.2 VLM GRPO损失函数

#### 3.2.1 基础PPO损失
VLM GRPO基于PPO的裁剪损失：
$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

#### 3.2.2 VLM GRPO改进
在PPO基础上，VLM GRPO添加了KL散度约束：
$$
L^{VLM-GRPO}(\theta) = L^{CLIP}(\theta) + \lambda_{KL} \cdot D_{KL}(\pi_\theta || \pi_{ref})
$$

其中：
- $L^{CLIP}(\theta)$：PPO裁剪损失
- $D_{KL}(\pi_\theta || \pi_{ref})$：与参考模型的KL散度
- $\lambda_{KL}$：KL散度权重（默认0.01）

## 4. 训练目标与核心原理

### 4.1 VLM GRPO到底在训练什么？

**这是理解VLM GRPO的关键问题！** VLM GRPO的目标是让模型学会从**文本和图像**中准确提取产品尺寸，通过组内相对比较来优化多模态理解能力。

#### 🎯 **训练目标**
```python
# 传统监督学习的目标
目标：让模型输出 = 标准尺寸
损失：MSELoss(模型输出, 标准尺寸)

# VLM GRPO的目标  
目标：让模型在组内比较中生成更准确的尺寸
损失：VLM_GRPOLoss(基于尺寸排序奖励和KL散度约束)
```

#### 🔄 **训练循环的本质**

每一轮训练都在回答这个问题：**"如何调整模型参数，让它从这个产品的文本和图片中提取更准确的尺寸？"**

```python
# 训练前：模型对产品可能生成
候选1: {"length": 20, "width": 15, "height": 25} → 奖励: 0.1 (偏差大)
候选2: {"length": 12, "width": 8, "height": 15}  → 奖励: 0.9 (接近真实)
候选3: {"length": 0, "width": 0, "height": 0}    → 奖励: 0.0 (完全错误)

# 训练后：模型对产品倾向于生成  
候选1: {"length": 12, "width": 8, "height": 15}  → 奖励: 0.9 (准确)
候选2: {"length": 11, "width": 7, "height": 14}  → 奖励: 0.8 (接近)
候选3: {"length": 13, "width": 9, "height": 16}  → 奖励: 0.7 (合理)
```

#### 📊 **多模态组内相对比较的优势**

1. **多模态融合**：同时利用文本和图像信息
2. **尺寸排序**：通过排序减少尺寸顺序的影响
3. **相对评估**：通过组内比较突出准确性差异
4. **鲁棒性**：对产品多样性有更好的适应性

### 4.2 核心训练流程

#### 4.2.1 多模态Rollout阶段
```python
def rollout(model, tokenizer, image_processor, text, images, ground_truth, num_rollouts=12):
    # 1. 构建多模态提示词
    prompt_str = build_prompt(text, images)
    user_msgs = [{"role": "user", "content": prompt_str, "images": images}]
    
    # 2. 生成多个候选尺寸
    for i in range(num_rollouts):
        answer = model.chat(msgs=user_msgs, tokenizer=tokenizer, ...)
        
        # 3. 计算尺寸排序奖励
        reward = calculate_reward(answer, ground_truth)
        rewards.append(reward)
```

#### 4.2.2 尺寸排序奖励计算
```python
def calculate_reward(answer, sample_label):
    # 1. 解析JSON格式的尺寸
    answer_dict = json.loads(answer)
    pred_dims = [answer_dict['length'], answer_dict['width'], answer_dict['height']]
    
    # 2. 排序处理
    label_sorted = sorted(label_dims, reverse=True)
    pred_sorted = sorted(pred_dims, reverse=True)
    
    # 3. 计算相对误差
    for i in range(min(len(label_sorted), len(pred_sorted))):
        rel_error = abs(pred_sorted[i] - label_sorted[i]) / label_sorted[i]
        total_error += rel_error
    
    # 4. 奖励 = 1 - 平均相对误差
    reward = max(0.0, 1.0 - total_error / min_len)
    return reward
```

#### 4.2.3 损失计算
```python
class VLM_GRPOLoss(nn.Module):
    def forward(self, log_probs, experience):
        # PPO裁剪损失
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
        ppo_loss = -torch.min(surr1, surr2)
        
        # KL散度约束
        kl = approx_kl_divergence(log_probs, log_probs_ref, action_mask)
        
        # 总损失
        total_loss = ppo_loss + kl_weight * kl
        return total_loss, kl.mean()
```

## 5. 代码结构详解

### 5.1 核心文件结构

```
learnings/dim_grpo/
├── train_vlm_parallel.py    # 多GPU分布式VLM GRPO训练脚本
├── loss.py                  # VLM GRPO损失函数实现
├── replay_buffer.py         # 经验回放缓冲区
├── run.py                  # 训练前后模型效果对比脚本
├── test_model.py           # 简化的单模型测试脚本
├── image_cache_new/        # 图片缓存目录
└── SoulTable_三品类尺寸待标注数据_标注测试0326_p12.xlsx  # 训练数据
```

### 5.2 关键函数解析

#### 5.2.1 `load_minicpm()` - 模型加载函数
```python
def load_minicpm(model_path):
    """
    加载MiniCPM多模态模型
    
    参数:
    - model_path: 模型路径
    
    返回:
    - model: 策略模型
    - tokenizer: 分词器
    - image_processor: 图像处理器
    """
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda:0"
    )
```

#### 5.2.2 `calculate_reward()` - 奖励计算函数
```python
def calculate_reward(answer, sample_label):
    """
    基于尺寸排序计算奖励
    
    参数:
    - answer: 模型预测的JSON格式尺寸
    - sample_label: 真实尺寸标签
    
    返回:
    - reward: 奖励分数 (0-1)
    """
```

#### 5.2.3 `build_prompt()` - 提示词构建函数
```python
def build_prompt(input_text, images):
    """
    构建多模态提示词
    
    参数:
    - input_text: 产品文本描述
    - images: 产品图片列表
    
    返回:
    - prompt: 格式化的提示词
    """
```

### 5.3 训练配置参数

```python
# 模型配置
model_path = "/data2/users/yszhang/quickllm/models/OpenBMB/MiniCPM-o-2_6"
checkpoint_path = "/data2/users/yszhang/quickllm/outputs/dim_grpo_minicpm_final_correct/model_step100.pt"

# 数据配置
data_path = "data.xlsx"
image_cache_dir = "image_cache_new"

# 训练超参数
batch_size = 16  # 训练批次大小
group_size = 12  # 每组候选答案数量
lr = 5e-6  # 学习率
kl_weight = 0.01  # KL散度权重
clip_eps = 0.2  # PPO裁剪参数

# 生成参数
max_length = 64  # 最大生成长度
temperature = 0.7  # 采样温度
```

## 6. 生动形象的例子

### 6.1 产品尺寸提取示例

让我们通过一个具体的产品来理解VLM GRPO的工作原理：

#### 🎯 **产品信息**
- **文本描述**：红色陶瓷咖啡杯，容量350ml
- **产品图片**：[咖啡杯图片]
- **真实尺寸**：长度12cm，宽度8cm，高度15cm

#### 📝 **训练过程**

**步骤1：生成候选尺寸**
```python
# 模型生成12个候选尺寸
候选1: {"length": 12, "width": 8, "height": 15}   # 完全正确
候选2: {"length": 11, "width": 7, "height": 14}   # 接近正确
候选3: {"length": 13, "width": 9, "height": 16}   # 略微偏差
候选4: {"length": 15, "width": 10, "height": 18}  # 偏差较大
候选5: {"length": 10, "width": 6, "height": 12}   # 偏小
候选6: {"length": 8, "width": 8, "height": 15}    # 宽度错误
候选7: {"length": 12, "width": 8, "height": 12}   # 高度错误
候选8: {"length": 0, "width": 0, "height": 0}     # 完全错误
候选9: {"length": 20, "width": 15, "height": 25}  # 严重偏差
候选10: {"length": 12, "width": 8, "height": 15}  # 重复正确
候选11: {"length": 11, "width": 8, "height": 15}  # 长度略小
候选12: {"length": 12, "width": 7, "height": 15}  # 宽度略小
```

**步骤2：尺寸排序处理**
```python
真实尺寸排序: [15, 12, 8]  # 高度>长度>宽度

候选1排序: [15, 12, 8]     # 完全匹配
候选2排序: [14, 11, 7]     # 相对误差小
候选3排序: [16, 13, 9]     # 相对误差小
候选4排序: [18, 15, 10]    # 相对误差大
候选5排序: [12, 10, 6]     # 相对误差大
候选6排序: [15, 8, 8]      # 宽度错误
候选7排序: [12, 12, 8]     # 高度错误
候选8排序: [0, 0, 0]       # 完全错误
候选9排序: [25, 20, 15]    # 严重偏差
候选10排序: [15, 12, 8]    # 完全匹配
候选11排序: [15, 11, 8]    # 长度略小
候选12排序: [15, 12, 7]    # 宽度略小
```

**步骤3：奖励计算**
```python
# 计算每个候选的相对误差
候选1: 误差 = 0.0, 奖励 = 1.0    # 完美
候选2: 误差 = 0.1, 奖励 = 0.9    # 优秀
候选3: 误差 = 0.1, 奖励 = 0.9    # 优秀
候选4: 误差 = 0.3, 奖励 = 0.7    # 良好
候选5: 误差 = 0.4, 奖励 = 0.6    # 一般
候选6: 误差 = 0.5, 奖励 = 0.5    # 较差
候选7: 误差 = 0.4, 奖励 = 0.6    # 一般
候选8: 误差 = 1.0, 奖励 = 0.0    # 错误
候选9: 误差 = 0.8, 奖励 = 0.2    # 很差
候选10: 误差 = 0.0, 奖励 = 1.0   # 完美
候选11: 误差 = 0.1, 奖励 = 0.9   # 优秀
候选12: 误差 = 0.1, 奖励 = 0.9   # 优秀
```

**步骤4：优势标准化**
```python
# 原始奖励
rewards = [1.0, 0.9, 0.9, 0.7, 0.6, 0.5, 0.6, 0.0, 0.2, 1.0, 0.9, 0.9]

# 标准化后的优势
advantages = (rewards - rewards.mean()) / rewards.std()
# 结果：准确答案优势为正，错误答案优势为负
```

### 6.2 训练效果对比

#### 🎯 **训练前 vs 训练后**

**训练前模型表现：**
```python
产品: "红色陶瓷咖啡杯，容量350ml"
图片: [咖啡杯图片]

候选尺寸:
- {"length": 20, "width": 15, "height": 25} (奖励: 0.2)  # 40%概率
- {"length": 0, "width": 0, "height": 0} (奖励: 0.0)     # 30%概率
- {"length": 12, "width": 8, "height": 15} (奖励: 1.0)   # 20%概率
- {"length": 10, "width": 6, "height": 12} (奖励: 0.6)   # 10%概率
```

**训练后模型表现：**
```python
产品: "红色陶瓷咖啡杯，容量350ml"
图片: [咖啡杯图片]

候选尺寸:
- {"length": 12, "width": 8, "height": 15} (奖励: 1.0)   # 60%概率 ↑
- {"length": 11, "width": 7, "height": 14} (奖励: 0.9)   # 25%概率 ↑
- {"length": 13, "width": 9, "height": 16} (奖励: 0.9)   # 10%概率 ↑
- {"length": 10, "width": 6, "height": 12} (奖励: 0.6)   # 3%概率 ↓
- {"length": 0, "width": 0, "height": 0} (奖励: 0.0)     # 2%概率 ↓
```

### 6.3 多模态理解示例

#### 🖼️ **图像+文本联合理解**

VLM GRPO特别适合需要多模态理解的产品尺寸提取：

```python
# 输入信息
文本: "这是一款便携式蓝牙音箱，支持防水功能"
图片: [蓝牙音箱产品图片]

# 模型推理过程
<think>
1. 从图片中观察到音箱的立体形状
2. 结合文本描述"便携式"，判断尺寸应该较小
3. 根据音箱的常见比例，推断长宽高关系
4. 考虑防水功能，外壳可能有一定厚度
</think>

# 输出结果
{"length": 8, "width": 6, "height": 4, "unit": "cm"}
```

这种多模态理解展示了：
1. **视觉感知**：从图片中提取形状和比例信息
2. **文本理解**：从描述中获取产品类型和特征
3. **知识融合**：结合视觉和文本信息进行推理
4. **尺寸估计**：基于多模态信息准确估计尺寸

## 7. 使用指南

### 7.1 环境准备

```bash
# 安装依赖
pip install torch transformers pandas pillow loguru openpyxl

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 使用GPU 0
```

### 7.2 数据准备

```bash
# 准备训练数据
# 确保以下文件存在：
# - SoulTable_三品类尺寸待标注数据_标注测试0326_p12.xlsx
# - image_cache_new/ (图片缓存目录)

# 数据格式要求：
# - text: 产品文本描述
# - label: 真实尺寸标签 (格式: "12 8 15")
# - sku_img, spu_img_1, spu_img_2, spu_img_3, spu_img_4: 图片URL
```

### 7.3 训练模型

```bash
cd learnings/dim_grpo

# 开始训练
python train_vlm_parallel.py

# 训练参数说明
# - gpu_ids: 使用的GPU ID列表
# - batch_size: 每个GPU的批量大小
# - group_size: 每个样本的生成数量
# - lr: 学习率
# - output_dir: 输出目录
```

### 7.4 测试模型

```bash
# 测试单个模型
python test_model.py

# 对比训练前后效果
python run.py

# 启动推理服务（如果有）
cd app
./run.sh start
```

### 7.5 监控训练

```bash
# 查看TensorBoard日志
tensorboard --logdir=runs/vlm_grpo_parallel --port=6006

# 查看训练指标
# - 总损失 (total_loss)
# - KL散度 (kl_divergence)  
# - 梯度范数 (grad_norm)
# - 平均奖励 (returns)
# - 成功率 (success_rate)
```

## 8. 性能优化建议

### 8.1 超参数调优

```python
# 推荐配置
lr = 5e-6              # 学习率不宜过大
kl_weight = 0.01       # KL散度权重适中
clip_eps = 0.2         # PPO裁剪参数
group_size = 12        # 组大小适中，平衡效果和效率
temperature = 0.7      # 保持适度的多样性
```

### 8.2 内存优化

```python
# 启用梯度检查点
model.gradient_checkpointing_enable()

# 使用混合精度训练
torch_dtype=torch.bfloat16

# 定期清理缓存
torch.cuda.empty_cache()

# 使用低CPU内存模式
low_cpu_mem_usage=True
```

### 8.3 多GPU训练

```python
# 分布式训练配置
gpu_ids = [0, 1, 2, 3]  # 使用多个GPU
batch_size_per_gpu = 4   # 每个GPU的批次大小

# 数据并行
model = torch.nn.DataParallel(model, device_ids=gpu_ids)
```

### 8.4 训练稳定性

```python
# 梯度裁剪
clip_grad_norm_(model.parameters(), max_norm=1.0)

# 损失检查
if not loss.isfinite():
    print(f"Loss not finite, skipping backward")
    continue

# 异常处理
try:
    answer = model.chat(...)
except Exception as e:
    logger.warning(f"预测失败: {e}")
    answer = default_answer
```

## 9. 常见问题与解决方案

### 9.1 训练不收敛

**问题**：模型训练后尺寸提取精度没有提升

**解决方案**：
1. 检查学习率是否过大
2. 调整KL散度权重
3. 增加组大小以提供更多对比信息
4. 检查奖励函数设计是否合理
5. 确保图片质量良好

### 9.2 内存不足

**问题**：GPU内存溢出

**解决方案**：
1. 减小batch_size
2. 减小group_size
3. 启用梯度检查点
4. 使用更小的模型
5. 减少图片分辨率

### 9.3 图片加载失败

**问题**：图片无法加载或下载失败

**解决方案**：
1. 检查图片URL是否有效
2. 使用图片缓存机制
3. 添加图片下载重试逻辑
4. 使用占位符图片进行测试

### 9.4 尺寸提取不准确

**问题**：模型提取的尺寸与实际偏差较大

**解决方案**：
1. 增加训练数据多样性
2. 调整奖励函数的权重
3. 检查数据标注质量
4. 增加图片数量和角度

### 9.5 多模态融合效果差

**问题**：模型无法有效融合文本和图像信息

**解决方案**：
1. 确保文本描述详细准确
2. 增加图片质量和数量
3. 调整提示词模板
4. 检查模型的多模态能力

## 10. 总结

VLM GRPO算法通过**多模态组内相对比较**机制，有效解决了产品尺寸提取任务中的挑战：

### 🎯 **核心优势**
1. **多模态融合**：同时利用文本和图像信息
2. **尺寸排序**：通过排序减少尺寸顺序的影响
3. **相对评估**：通过组内比较突出准确性差异
4. **鲁棒性**：对产品多样性有更好的适应性

### 🔄 **训练流程**
1. **多模态Rollout阶段**：对每个产品生成多个候选尺寸
2. **尺寸排序奖励**：基于排序的相对误差计算奖励
3. **优势标准化**：对组内奖励进行标准化
4. **模型更新**：通过VLM GRPO损失更新模型参数

### 📈 **应用场景**
- 电商产品尺寸提取
- 工业产品测量
- 家具尺寸估计
- 包装设计优化
- 物流体积计算

### 🚀 **技术特色**
- **多模态理解**：文本+图像联合推理
- **尺寸排序**：消除尺寸顺序影响
- **相对奖励**：基于组内比较的精细评估
- **分布式训练**：支持多GPU并行训练

VLM GRPO为多模态产品尺寸提取提供了一种有效的方法，通过多模态组内相对比较机制，能够更好地指导模型学习准确的产品尺寸提取能力。 