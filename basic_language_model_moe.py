# -*- coding: utf-8 -*- 
# @Time : 2023/12/13 02:09 
# @Author : ys 
# @File : basic_language_model_moe.py

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