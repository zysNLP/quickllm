# GRPO (Group Relative Policy Optimization) 算法详解与实现指南

本文代码参见：[https://github.com/zysNLP/quickllm/tree/main/learnings/tiny-grpo](https://github.com/zysNLP/quickllm/tree/main/learnings/tiny-grpo)；感谢star。本文内容生动形象、但也非常长非常详细，请参照代码逐行耐心查看

## 1. GRPO算法简介

组相对策略优化（Group Relative Policy Optimization, GRPO）是一种基于PPO改进的强化学习算法，专门用于数学推理任务的优化。与传统的PPO不同，GRPO引入了**组内相对比较**的概念，通过在同一问题下生成多个候选答案并进行相对评估，来更好地指导模型学习。

### 🎯 **为什么需要GRPO？**

在数学推理任务中，传统的PPO面临以下挑战：
1. **答案多样性**：同一个数学问题可能有多种正确的解题思路
2. **部分正确性**：答案可能部分正确，需要更精细的奖励机制
3. **推理过程重要性**：不仅要答案正确，推理过程也要合理

GRPO通过**组内相对比较**解决了这些问题：

```python
# 传统PPO：单个答案 vs 标准答案
问题: "1+1等于多少？"
模型答案: "1+1等于2" → 奖励: +1.0 (完全正确)

# GRPO：多个候选答案的相对比较
问题: "1+1等于多少？"
候选1: "1+1等于2"     → 奖励: +1.0 (最佳)
候选2: "1+1等于2"     → 奖励: +0.8 (重复但正确)  
候选3: "1+1等于3"     → 奖励: +0.01 (错误)
候选4: "这是数学问题"  → 奖励: +0.01 (无关)
```

## 2. 核心概念：3+2理解法

### 2.1 三个模型

#### 2.1.1 策略模型（Actor Model）
- **作用**：待优化的主模型，负责生成数学推理答案
- **参数更新**：✅ 参与训练，通过GRPO损失进行优化
- **代码位置**：`model = AutoModelForCausalLM.from_pretrained(...)`

#### 2.1.2 参考模型（Reference Model）
- **作用**：防止策略模型偏离原始模型太远，提供KL散度约束
- **参数更新**：❌ 不参与训练，权重固定
- **代码位置**：`reference_model = AutoModelForCausalLM.from_pretrained(...)`

#### 2.1.3 奖励函数（Reward Function）
- **作用**：评估生成答案的质量，提供相对奖励信号
- **实现方式**：基于答案匹配的启发式函数
- **代码位置**：`rollout()`函数中的奖励计算逻辑

### 2.2 两个核心机制

#### 2.2.1 组内相对比较（Group Relative Comparison）
```python
# 在rollout()函数中实现
def rollout(model, tokenizer, task, oracle_answer, num_rollouts=12):
    # 对同一个问题生成多个候选答案
    completions = model.generate(...)  # 生成12个候选答案
    
    # 计算每个候选答案的相对奖励
    for i, completion in enumerate(completions):
        if answer == oracle_answer:
            reward = 1.0      # 完全正确
        elif oracle_answer in answer:
            reward = 0.5      # 部分正确
        else:
            reward = 0.01     # 错误答案
```

#### 2.2.2 优势标准化（Advantage Normalization）
```python
def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # 对组内奖励进行标准化，突出相对差异
    return (returns - returns.mean()) / (returns.std() + eps)
```

## 3. 数学推导过程

### 3.1 基础概念

#### 3.1.1 组内轨迹
在GRPO中：
- **组**：对同一个问题生成的多个候选答案
- **轨迹**：一次完整的答案生成过程
- **状态**：当前的推理上下文
- **动作**：生成下一个推理步骤

组内轨迹定义：
$$
G = \{\tau_1, \tau_2, \ldots, \tau_n\}
$$

其中每个$\tau_i$是对同一问题的不同候选答案。

#### 3.1.2 相对奖励
GRPO的核心创新在于相对奖励机制：
$$
R_{relative}(\tau_i) = R_{absolute}(\tau_i) - \bar{R}_{group}
$$

其中$\bar{R}_{group}$是组内平均奖励。

### 3.2 GRPO损失函数

#### 3.2.1 基础PPO损失
GRPO基于PPO的裁剪损失：
$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

#### 3.2.2 GRPO改进
在PPO基础上，GRPO添加了KL散度约束：
$$
L^{GRPO}(\theta) = L^{CLIP}(\theta) + \lambda_{KL} \cdot D_{KL}(\pi_\theta || \pi_{ref})
$$

其中：
- $L^{CLIP}(\theta)$：PPO裁剪损失
- $D_{KL}(\pi_\theta || \pi_{ref})$：与参考模型的KL散度
- $\lambda_{KL}$：KL散度权重（默认0.01）

## 4. 训练目标与核心原理

### 4.1 GRPO到底在训练什么？

**这是理解GRPO的关键问题！** GRPO的目标是让模型学会生成**更高质量的数学推理答案**，通过组内相对比较来优化策略。

#### 🎯 **训练目标**
```python
# 传统监督学习的目标
目标：让模型输出 = 标准答案
损失：CrossEntropyLoss(模型输出, 标准答案)

# GRPO的目标  
目标：让模型在组内比较中生成更好的答案
损失：GRPOLoss(基于组内相对奖励和KL散度约束)
```

#### 🔄 **训练循环的本质**

每一轮训练都在回答这个问题：**"如何调整模型参数，让它在这个问题下生成更好的候选答案？"**

```python
# 训练前：模型对"1+1等于多少？"可能生成
候选1: "1+1等于3"     # 奖励: 0.01 (错误)
候选2: "这是数学问题"  # 奖励: 0.01 (无关)
候选3: "1+1等于2"     # 奖励: 1.0  (正确)

# 训练后：模型对"1+1等于多少？"倾向于生成  
候选1: "1+1等于2"     # 奖励: 1.0  (正确)
候选2: "1+1=2"        # 奖励: 0.5  (部分正确)
候选3: "答案是2"      # 奖励: 0.5  (部分正确)
```

#### 📊 **组内相对比较的优势**

1. **多样性探索**：生成多个候选答案，探索不同的解题思路
2. **相对评估**：通过组内比较，突出答案质量的相对差异
3. **稳定性提升**：减少单个答案的偶然性影响
4. **推理质量**：不仅关注答案正确性，还关注推理过程

### 4.2 核心训练流程

#### 4.2.1 Rollout阶段
```python
def rollout(model, tokenizer, task, oracle_answer, num_rollouts=12):
    # 1. 构建提示词
    chat_prompt = tokenizer.apply_chat_template(chat_messages, ...)
    
    # 2. 生成多个候选答案
    sequence_ids = model.generate(
        input_ids=model_inputs["input_ids"],
        generation_config=generation_config
    )
    
    # 3. 计算相对奖励
    for i, completion in enumerate(completions):
        answer_match = re.search(r"<answer>(.*?)</answer>", completion)
        if answer_match:
            answer = answer_match.group(1)
            if answer == oracle_answer:
                reward = 1.0      # 完全正确
            elif oracle_answer in answer:
                reward = 0.5      # 部分正确
            else:
                reward = 0.01     # 错误
```

#### 4.2.2 优势计算
```python
# 对组内奖励进行标准化
advantages = group_advantages(returns)
# advantages = (returns - returns.mean()) / (returns.std() + eps)
```

#### 4.2.3 损失计算
```python
class GRPOLoss(nn.Module):
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
learnings/tiny-grpo/
├── train.py              # 主训练脚本
├── loss.py               # GRPO损失函数实现
├── replay_buffer.py      # 经验回放缓冲区
├── run.py               # 推理测试脚本
└── app/                 # 推理服务应用
    ├── config.py        # 配置文件
    ├── models.py        # 数据模型
    ├── utils.py         # 工具函数
    └── run.sh          # 服务管理脚本
```

### 5.2 关键函数解析

#### 5.2.1 `rollout()` - 核心采样函数
```python
@torch.no_grad()
def rollout(model, tokenizer, task, oracle_answer, num_rollouts=12):
    """
    对单个问题生成多个候选答案并计算奖励
    
    参数:
    - model: 策略模型
    - tokenizer: 分词器
    - task: 数学问题
    - oracle_answer: 标准答案
    - num_rollouts: 生成候选答案数量
    
    返回:
    - sequence_ids: 生成的序列ID
    - returns: 奖励值
    - action_mask: 动作掩码
    - completions: 生成的文本
    """
```

#### 5.2.2 `GRPOLoss` - 损失函数
```python
class GRPOLoss(nn.Module):
    """
    GRPO损失函数，结合PPO裁剪损失和KL散度约束
    
    参数:
    - clip_eps: PPO裁剪参数 (默认0.2)
    - kl_weight: KL散度权重 (默认0.01)
    """
```

#### 5.2.3 `ReplayBuffer` - 经验缓冲区
```python
class ReplayBuffer:
    """
    存储训练经验，支持批处理
    
    主要功能:
    - 存储Experience对象
    - 支持批处理采样
    - 自动清理过期数据
    """
```

### 5.3 训练配置参数

```python
# 训练超参数
seed = 42                    # 随机种子
train_batch_size = 16        # 训练批次大小
lr = 5e-6                    # 学习率
kl_weight = 0.01             # KL散度权重
clip_eps = 0.2               # PPO裁剪参数

# 采样参数
group_size = 12              # 每组候选答案数量
rollouts_per_step = 32       # 每步处理的问题数量
epochs_per_step = 1          # 每步训练轮数

# 生成参数
max_length = 1024            # 最大生成长度
temperature = 1.0            # 采样温度
top_p = 1.0                  # 核采样参数
```

## 6. 生动形象的例子

### 6.1 数学推理示例

让我们通过一个具体的数学问题来理解GRPO的工作原理：

#### 🎯 **问题**：小明有5个苹果，给了小红2个，又买了3个，现在有多少个苹果？

#### 📝 **训练过程**

**步骤1：生成候选答案**
```python
# 模型生成12个候选答案
候选1: "<think>小明原来有5个苹果，给了小红2个，剩下3个，又买了3个，所以现在有6个</think><answer>6个</answer>"
候选2: "<think>5-2+3=6</think><answer>6</answer>"
候选3: "<think>5个苹果，给了2个，买了3个，5-2+3=6</think><answer>6个苹果</answer>"
候选4: "<think>5个苹果减去2个等于3个，再加上3个等于6个</think><answer>6</answer>"
候选5: "<think>5-2=3, 3+3=6</think><answer>6</answer>"
候选6: "<think>小明有5个苹果</think><answer>5个</answer>"  # 推理不完整
候选7: "<think>这是一个数学问题</think><answer>需要计算</answer>"  # 无关答案
候选8: "<think>5+2+3=10</think><answer>10个</answer>"  # 计算错误
候选9: "<think>5个苹果</think><answer>5</answer>"  # 推理不完整
候选10: "<think>5-2+3=6</think><answer>6</answer>"
候选11: "<think>原来5个，给2个，买3个，5-2+3=6</think><answer>6个</answer>"
候选12: "<think>5个苹果给了2个，又买了3个，所以是5-2+3=6</think><answer>6个苹果</answer>"
```

**步骤2：计算奖励**
```python
标准答案: "6个"

# 奖励计算
候选1: 奖励 = 1.0   # 完全正确，推理完整
候选2: 奖励 = 0.5   # 答案正确，推理简单
候选3: 奖励 = 1.0   # 完全正确，推理清晰
候选4: 奖励 = 1.0   # 完全正确，推理详细
候选5: 奖励 = 0.5   # 答案正确，推理简洁
候选6: 奖励 = 0.01  # 推理不完整
候选7: 奖励 = 0.01  # 无关答案
候选8: 奖励 = 0.01  # 计算错误
候选9: 奖励 = 0.01  # 推理不完整
候选10: 奖励 = 0.5  # 答案正确，推理简洁
候选11: 奖励 = 1.0  # 完全正确，推理清晰
候选12: 奖励 = 1.0  # 完全正确，推理详细
```

**步骤3：优势标准化**
```python
# 原始奖励
returns = [1.0, 0.5, 1.0, 1.0, 0.5, 0.01, 0.01, 0.01, 0.01, 0.5, 1.0, 1.0]

# 标准化后的优势
advantages = (returns - returns.mean()) / returns.std()
# 结果：好的答案优势为正，差的答案优势为负
```

**步骤4：模型更新**
```python
# 增加生成高优势答案的概率
# 降低生成低优势答案的概率
# 保持与参考模型的KL散度约束
```

### 6.2 训练效果对比

#### 🎯 **训练前 vs 训练后**

**训练前模型表现：**
```python
问题: "一个长方形的长是8cm，宽是6cm，面积是多少？"

候选答案:
- "面积是48平方厘米" (奖励: 1.0)     # 30%概率
- "8×6=48" (奖励: 0.5)              # 25%概率  
- "需要计算面积" (奖励: 0.01)        # 20%概率
- "长方形面积公式" (奖励: 0.01)      # 15%概率
- "48cm²" (奖励: 0.5)               # 10%概率
```

**训练后模型表现：**
```python
问题: "一个长方形的长是8cm，宽是6cm，面积是多少？"

候选答案:
- "面积是48平方厘米" (奖励: 1.0)     # 60%概率 ↑
- "8×6=48平方厘米" (奖励: 1.0)      # 25%概率 ↑
- "长8cm，宽6cm，面积=8×6=48cm²" (奖励: 1.0)  # 10%概率 ↑
- "48平方厘米" (奖励: 0.5)          # 3%概率 ↓
- "需要计算" (奖励: 0.01)           # 2%概率 ↓
```

### 6.3 推理过程示例

#### 🧠 **思维链推理**

GRPO特别适合需要推理过程的数学问题：

```python
问题: "一个班级有30名学生，其中男生占60%，女生占40%，男生比女生多多少人？"

模型推理过程:
<think>
1. 首先计算男生人数：30 × 60% = 18人
2. 然后计算女生人数：30 × 40% = 12人  
3. 最后计算差值：18 - 12 = 6人
</think>
<answer>6人</answer>
```

这种推理过程展示了：
1. **步骤清晰**：每个计算步骤都有明确说明
2. **逻辑正确**：计算过程符合数学逻辑
3. **答案准确**：最终答案正确

## 7. 使用指南

### 7.1 环境准备

```bash
# 安装依赖
pip install torch transformers pandas wandb tensorboard

# 设置环境变量
export CUDA_VISIBLE_DEVICES=3  # 使用GPU 3
```

### 7.2 训练模型

```bash
cd learnings/tiny-grpo

# 开始训练
python train.py

# 训练参数说明
# - model_name: 基础模型路径
# - gsm8k_path: 训练数据路径
# - checkpoint_path: 模型保存路径
# - train_batch_size: 训练批次大小
# - group_size: 每组候选答案数量
```

### 7.3 推理测试

```bash
# 使用训练后的模型进行推理
python run.py

# 启动推理服务
cd app
./run.sh start
```

### 7.4 监控训练

```bash
# 查看TensorBoard日志
tensorboard --logdir ./runs/grpo_train

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
temperature = 1.0      # 保持多样性
```

### 8.2 内存优化

```python
# 启用梯度检查点
model.gradient_checkpointing_enable()

# 使用混合精度训练
torch_dtype=torch.bfloat16

# 定期清理缓存
torch.cuda.empty_cache()
```

### 8.3 训练稳定性

```python
# 梯度裁剪
clip_grad_norm_(model.parameters(), max_norm=1.0)

# 损失检查
if not loss.isfinite():
    print(f"Loss not finite, skipping backward")
    continue
```

## 9. 常见问题与解决方案

### 9.1 训练不收敛

**问题**：模型训练后性能没有提升

**解决方案**：
1. 检查学习率是否过大
2. 调整KL散度权重
3. 增加组大小以提供更多对比信息
4. 检查奖励函数设计是否合理

### 9.2 内存不足

**问题**：GPU内存溢出

**解决方案**：
1. 减小batch_size
2. 减小group_size
3. 启用梯度检查点
4. 使用更小的模型

### 9.3 推理质量下降

**问题**：训练后模型推理能力下降

**解决方案**：
1. 增加KL散度权重，防止偏离原始模型太远
2. 检查参考模型是否正确加载
3. 调整clip_eps参数
4. 增加训练数据多样性

## 10. 总结

GRPO算法通过**组内相对比较**机制，有效解决了数学推理任务中的挑战：

### 🎯 **核心优势**
1. **相对评估**：通过组内比较提供更精细的奖励信号
2. **多样性探索**：生成多个候选答案，探索不同解题思路
3. **稳定性提升**：减少单个答案的偶然性影响
4. **推理质量**：不仅关注答案正确性，还关注推理过程

### 🔄 **训练流程**
1. **Rollout阶段**：对每个问题生成多个候选答案
2. **奖励计算**：基于答案匹配计算相对奖励
3. **优势标准化**：对组内奖励进行标准化
4. **模型更新**：通过GRPO损失更新模型参数

### 📈 **应用场景**
- 数学推理任务
- 逻辑推理问题
- 需要推理过程的问答任务
- 多步骤计算问题

GRPO为数学推理模型的训练提供了一种有效的方法，通过组内相对比较机制，能够更好地指导模型学习高质量的推理过程。
