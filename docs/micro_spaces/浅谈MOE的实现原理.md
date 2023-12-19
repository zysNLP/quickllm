## 浅谈MOE的实现原理（一）

MOE的输入inputs张量的维度是[batch_size, seq_len, hidden_size]，输出也是一样的[batch_size, seq_len, hidden_size]，因此但从输出输出看，就可以大致把MOE的作为self-attention的另一种类似的形式了。

我也不好说，姑且看一下代码捋一下它的思路：

### 一、主函数：

```python
# -*- coding: utf-8 -*- 
# @Time : 2023/12/13 02:09 
# @File : basic_language_model_moe_3d.py

import torch
from torch import nn
from quickllm.layers.moe import MoE


if __name__ == "__main__":

    moe = MoE(
        dim=512,  # 输入张量的维度
        num_experts=16,  # 专家数量，可以增加该参数而不增加计算量
        hidden_dim=512 * 4,  # 每个专家网络中的隐藏层维度，默认为 4 倍输入维度
        activation=nn.LeakyReLU,  # 使用的激活函数，默认为 GELU
        second_policy_train='random',  # 使用的第二名专家的训练策略
        second_policy_eval='random',  # 使用的第二名专家的验证策略
        second_threshold_train=0.2,  # 训练时使用的第二名专家阈值
        second_threshold_eval=0.2,  # 测试时使用的第二名专家阈值
        capacity_factor_train=1.25,  # 每个专家网络在单个批次中的固定容量，需要额外的容量以防门控不平衡
        capacity_factor_eval=2.,  # capacity_factor_* 应设置为 >=1 的值
        loss_coef=1e-2  # 辅助专家平衡辅助损失的乘数
    )
    inputs = torch.randn(4, 1024, 512)
    out, aux_loss = moe(inputs)  # (4, 1024, 512), (1,)
    print(out.shape, aux_loss.shape)
```

没什么好说的，导包和定义参数，初始化基础类，然后把输入放进去，得一个输出。注意一下它的维度，正是我刚刚提到的

进到主函数中，来看一下它是怎么回事

```python
    def __init__(self,......):
        super().__init__()
        self.num_experts = num_experts

        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval,
                         'second_threshold_train': second_threshold_train,
                         'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train,
                         'capacity_factor_eval': capacity_factor_eval}
        self.gate = Top2Gating(dim, num_gates=num_experts, **gating_kwargs)
        self.experts = default(experts, lambda: Experts(dim, num_experts=num_experts, hidden_dim=hidden_dim,
                                                        activation=activation))
        self.loss_coef = loss_coef
```

By the way， 类的初始化这里加多了一些参数，gating_kwargs、gate、experts、loss，先不管是啥，kwargs看起来是初始参数的一个集合，另外的变量是调用别的类和方法了，只是从名字上看有个印象。哦，”门控参数？门控对象？专家对象？loss值？“，大概是这些意思

```python
    def forward(self, inputs, **kwargs):
        b, n, d, e = *inputs.shape, self.num_experts
        dispatch_tensor, combine_tensor, loss = self.gate(inputs)
        expert_inputs = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor)

        # Now feed the expert inputs through the experts.
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(e, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        output = torch.einsum('ebcd,bnec->bnd', expert_outputs, combine_tensor)
        return output, loss * self.loss_coef
```

forward函数传入的是刚刚的inputs和别的可能的参数，然后：1.把维度拆分了，2.计算几个tensor/gate，3.用结果计算一个专家输入expert_inputs，4.把它的shape取出来reshape一下，5.作用在self.experts上得一个outputs结果，6.最后再计算一次张量相乘，得最后的输出。值得注意是的，最后这步相乘用的是专家输出expert_outputs和第一次输出得到的combine_tensor。

那么我们就挨个来看，gate和experts函数

### 二、Gate函数

```
class Top2Gating(nn.Module):
    def __init__(。。。。):
        super().__init__()

        self.eps = eps
        self.num_gates = num_gates
        self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates))

        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval
        self.second_threshold_train = second_threshold_train
        self.second_threshold_eval = second_threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

    def forward(self, x, importance=None):
        *_, b, group_size, dim = x.shape
        num_gates = self.num_gates

     		。。。太长了
        return dispatch_tensor, combine_tensor, loss
```

对应Top2Gating类，又是一些初始化，后续初始化的工作我们不谈了，代码太长了我们分开来看

#### 2.1 gate函数第一部分

```
    def forward(self, x, importance=None):
        *_, b, group_size, dim = x.shape
        num_gates = self.num_gates

        if self.training:
            policy = self.second_policy_train
            threshold = self.second_threshold_train
            capacity_factor = self.capacity_factor_train
        else:...

        # 转换成一个"初始的门"
        raw_gates = torch.einsum('...bnd,...de->...bne', x, self.w_gating)
        raw_gates = raw_gates.softmax(dim=-1)  # 最后一维转为概率分布

        # FIND TOP 2 EXPERTS PER POSITON
        # Find the top expert for each position. shape=[batch, group]
        gate_1, index_1 = top1(raw_gates)
        mask_1 = F.one_hot(index_1, num_gates).float()   # top_k的index转为onehot=1
        density_1_proxy = raw_gates

        if importance is not None:
            equals_one_mask = (importance == 1.).float()
            mask_1 *= equals_one_mask[..., None]
            gate_1 *= equals_one_mask
            density_1_proxy = density_1_proxy * equals_one_mask[..., None]
            del equals_one_mask

        gates_without_top_1 = raw_gates * (1. - mask_1)  # raw_gates中mask=1的位置设置为0

        gate_2, index_2 = top1(gates_without_top_1)
        mask_2 = F.one_hot(index_2, num_gates).float()

        if importance is not None:
            ......
        
        # normalize top2 gate scores， gate1和2按具体值在对应位置加权平均归一
        denom = gate_1 + gate_2 + self.eps
        gate_1 /= denom
        gate_2 /= denom
```

拿到x的每一个shape后，如果模式是训练模式，就定义几个参数policy  threshold  capacity_factor，注意，第一步，是x和一个self.w_gating相乘，这个向量的维度是(hidden_size, num_experts)，用[batch_size, seq_len, hidden_size]维度的输入与之相乘，得到的结果是[batch_size, seq_len, num_gates]，self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_experts))是这么来的，姑且把最后一个维度因此最后一个维度我们称作是”专家数量“

因此我们得到了batch组数据，每个数据的每个token都有num_experts个专家

再做一个softmax把它转换成概率分布，然后传进一个top1函数，如注释所言，得到最终的raw_gates变量，维度还是 (batch_size, seq_len)：

```
def top1(t):
    values, index = t.topk(k=1, dim=-1)   # 最后一个维度取最大
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))  # 最大的那个值的索引
    return values, index
```

很明显，因为最大值只有一个，所以最后一维就不需要了

之后进了一个ont-hot函数，最终结果得一个mask1变量，具体实现不谈了，总之得到的仍然是和input相同的维度，但因为是通过index1和num_gates得来的，注意，代码里的num_gates=我们说的num_experts。因此这个mask1矩阵在index1有值的位置取值为1，其他取值为0

后续再把raw_gates与1-mask1的结果相乘，得一个经过mask之后的raw_gates

之后重复第二个步骤，得到另外两个gate_2, index_2和mask2。同时给raw_gates变量换个名字叫density_1_proxy

然后再看denom = 后面三行，显然是做了一个加权求和，只是这是对gate_1和gate_2做的，我们知道gate_1/2都是从top1函数来的。。。

是不是到这里有点晕了，我因为方便大家查看起见，很多代码都已经省略掉了。建议大家还是自己从git上搜quickllm下载项目自己调试。总之，这一步大概意思是，对得到的两个专家变量，各自做一个top取值，完了再来一次加权最后得加权后的top取值。

继续来看：

```
        density_1 = mask_1.mean(dim=-2)
        # Something continuous that is correlated with what we want to equalize.
        density_1_proxy = density_1_proxy.mean(dim=-2)
        loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)
```

#### 2.2 gate函数第二部分

给mask1做平均得一个density_1，给density_1_proxy=raw-inputs做一个平均得density_1_proxy，最后把他们俩来个cross函数

```
 				# Depending on the policy in the hparams, we may drop out some of the
        # second-place experts.
        if policy == "all":
            pass
        elif policy == "none":
            mask_2 = torch.zeros_like(mask_2)
        elif policy == "threshold":
            mask_2 *= (gate_2 > threshold).float()
        elif policy == "random":
            probs = torch.zeros_like(gate_2).uniform_(0., 1.)
            mask_2 *= (probs < (gate_2 / max(threshold, self.eps))).float().unsqueeze(-1)
        else:
            raise ValueError(f"Unknown policy {policy}")

        # Each sequence sends (at most?) expert_capacity positions to each expert.
        # Static expert_capacity dimension is needed for expert batch sizes
        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)
        
        position_in_expert_1 = cumsum_exclusive(mask_1, dim=-2) * mask_1
        # Remove the elements that don't fit. [batch, group, experts]
        mask_1 *= (position_in_expert_1 < expert_capacity_f).float()
        # [batch, experts]
        # How many examples in this sequence go to this expert
        mask_1_count = mask_1.sum(dim=-2, keepdim=True)
        # [batch, group] - mostly ones, but zeros where something didn't fit
        mask_1_flat = mask_1.sum(dim=-1)
        # [batch, group]
        position_in_expert_1 = position_in_expert_1.sum(dim=-1)
        # Weight assigned to first expert.  [batch, group]
        gate_1 *= mask_1_flat
```

之后是根据policy的不同选择一个mask2的方式，简单起见我们mask_2 = torch.zeros_like(mask_2)这个最直观的为例。

定义”专家的能力“，好家伙，等于group_size和另一个只的最小值。初步来看，group_size=seq-len，也就理解了，一个专家到底能把握多少”seq_len"。

```
def cumsum_exclusive(t, dim=-1):
    num_pad_dims = - dim - 1
    pre_padding = (0, 0) * num_pad_dims
    pre_slice = (slice(None),) * num_pad_dims
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded_t[(..., slice(None, -1), *pre_slice)]
```

后面又对mask做了一些cumsum_exclusive，不知道干嘛，看上去是padding和累加？又来一个截断再后面是对mask做一些求和，得到各自几个变量

脑子累，先不谈了，下节继续。

参考代码链接1：[moe](https://github.com/zysNLP/quickllm/blob/main/basic_language_model_moe.py)

参考代码链接2：[Mistal AI SRC](https://github.com/lucidrains/mixture-of-experts)

