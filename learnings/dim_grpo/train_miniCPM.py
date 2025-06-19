import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# 在文件顶部，确保导入了 AutoImageProcessor
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
from PIL import Image
from pathlib import Path
from loss import GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch
import numpy as np
from loguru import logger
import torch.nn.functional as F
import hashlib
from torch.nn.utils.rnn import pad_sequence
from transformers import GenerationConfig
from dataclasses import dataclass, fields
from typing import Optional
import torch.nn as nn
import json
import re

# 设置环境变量，确保只使用指定的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

def get_md5(url):
    """获取URL的MD5值作为文件名"""
    # 确保URL是字符串类型
    if not isinstance(url, str):
        url = str(url)
    return hashlib.md5(url.encode()).hexdigest()

# 数据加载
class DimDataset(torch.utils.data.Dataset):
    def __init__(self, excel_path, image_cache_dir):
        self.df = pd.read_excel(excel_path)
        self.image_cache_dir = Path(image_cache_dir)
        self.fields = ["sku_img", "spu_img_1", "spu_img_2", "spu_img_3", "spu_img_4"]
        logger.info(f"加载数据集: {excel_path}, 共{len(self.df)}条")
        
        # 优化图片查找：预先构建一个文件名集合
        self.existing_images = {f.name for f in self.image_cache_dir.glob('*') if f.is_file()}
        logger.info(f"找到 {len(self.existing_images)} 个图片文件")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_paths = []
        for f in self.fields:
            url = row[f]
            if isinstance(url, str) and url.strip():
                # 从URL中提取文件名，通常是最后一个/之后的部分
                url_filename_part = url.split('/')[-1]
                # 使用URL的MD5值 + 后缀作为本地文件名
                fname = get_md5(url) + os.path.splitext(url_filename_part)[1]
                if fname in self.existing_images:
                    img_paths.append(self.image_cache_dir / fname)
                else:
                    # 降级警告为调试信息，避免刷屏
                    # logger.warning(f"图片不存在: {self.image_cache_dir / fname}")
                    pass # 忽略不存在的图片
        
        # 确保至少有一张图片，否则跳过这个样本可能更好，但这里我们先打开存在的
        images = [Image.open(p).convert('RGB') for p in img_paths if os.path.exists(p)]
        text = str(row["text"])
        # 标签处理保持不变
        label = str(row["label"]).replace("[", "").replace("]", "").replace(",", " ").replace("cm", "").strip()
        return {
            "images": images,
            "text": text,
            "label": label
        }

def collate_fn(batch):
    # 过滤掉没有有效图片的样本
    valid_batch = [item for item in batch if item['images']]
    if not valid_batch:
        return None
    
    batch_images = [item["images"] for item in valid_batch]
    batch_text = [item["text"] for item in valid_batch]
    batch_label = [item["label"] for item in valid_batch]
    return {"images": batch_images, "text": batch_text, "label": batch_label}

def load_minicpm(model_path):
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,  # 启用低CPU内存使用
        device_map="cuda:0"  # 改成使用gpu 0号卡
    )
    
    # 启用梯度检查点以节省内存
    model.gradient_checkpointing_enable()
    
    # 将模型移到GPU
    model = model.cuda()
    model.eval()
    
    # 设置模型参数为不需要梯度（除了需要训练的部分）
    for param in model.parameters():
        param.requires_grad = False
    
    # 只对特定层启用梯度
    for name, param in model.named_parameters():
        if "lm_head" in name or "embed_tokens" in name:
            param.requires_grad = True
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # --- 这是关键的、正确的补充 ---
    # 显式加载与模型配套的图像处理器
    image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 返回所有三个组件
    return model, tokenizer, image_processor

# 删除了 get_log_probs 函数，因为我们将使用更底层的logits

def build_prompt(input_text, images):
    # 简化prompt，移除动态图像数量字符串，因为模型通过<image> token识别
    prompt = (
        f"基于以下产品信息，包括文本和多张图片，提取产品的长、宽、高尺寸。\n"
        f"产品文本描述: {input_text}\n"
        "请严格按照以下JSON格式输出，不要包含任何额外说明或markdown标记：\n"
        "{\"length\": 数字, \"width\": 数字, \"height\": 数字, \"unit\": \"cm\"}"
    )
    return prompt

# 删除了 rollout_and_forward，逻辑将合并到主循环中
# 删除了 sequences_log_probs，这是错误的核心，将被完全替换

def masked_mean(values, mask, dim=-1):
    """计算带掩码的平均值"""
    return (values * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1.0)

def approx_kl_divergence(log_probs, log_probs_ref, action_mask):
    """计算近似的KL散度，使用更稳定的方法"""
    # 计算log_ratio
    log_ratio = log_probs - log_probs_ref
    
    # 添加调试信息
    logger.info(f"KL调试 - log_probs: {log_probs.item():.6f}, log_probs_ref: {log_probs_ref.item():.6f}")
    logger.info(f"KL调试 - log_ratio: {log_ratio.item():.6f}")
    
    # 检查数值是否合理
    if torch.abs(log_ratio) > 50:  # 降低阈值，更保守
        logger.warning(f"log_ratio异常大: {log_ratio.item():.6f}，可能存在数值问题")
        # 如果log_ratio异常大，使用更保守的计算
        if log_ratio > 0:
            kl = torch.tensor(5.0, device=log_ratio.device)  # 更保守的估计
        else:
            kl = torch.tensor(0.05, device=log_ratio.device)   # 更保守的估计
    else:
        # 使用更稳定的KL散度计算
        # 对于较小的log_ratio，使用标准公式
        if torch.abs(log_ratio) <= 3.0:  # 降低阈值
            kl = (log_ratio.exp() - 1 - log_ratio)
        else:
            # 对于较大的log_ratio，使用近似公式
            kl = log_ratio + 0.5 * log_ratio.pow(2) * torch.exp(-torch.abs(log_ratio))
    
    # 应用action_mask
    kl = kl * action_mask
    
    # 添加调试信息
    logger.info(f"KL调试 - 最终KL: {kl.item():.6f}")
    
    # 如果KL值过大，给出警告但保留原始值
    if kl.item() > 50:  # 降低阈值
        logger.warning(f"KL散度值过大: {kl.item():.6f}，可能需要检查模型输出")
    
    return kl

# --- 新增：GRPO Loss的修改 ---
# 确保 GRPOLoss 能够处理我们新的输入格式
class GRPOLoss(nn.Module):
    """GRPO actor loss"""

    def __init__(self, clip_eps: float, kl_weight: float) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight

    def forward(
        self,
        log_probs: torch.Tensor,
        log_probs_old: torch.Tensor,
        log_probs_ref: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 计算 KL 散度
        kl = approx_kl_divergence(
            log_probs=log_probs,
            log_probs_ref=log_probs_ref,
            action_mask=action_mask,
        )

        # 计算重要性采样比率，添加数值稳定性
        log_ratio = log_probs - log_probs_old
        # 使用更合理的范围限制
        log_ratio = torch.clamp(log_ratio, min=-3.0, max=3.0)  # 更保守的范围
        ratio = log_ratio.exp()
        
        # 计算裁剪后的损失
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2) + self.kl_weight * kl
        
        # 限制loss的范围，但使用更合理的值
        loss = torch.clamp(loss, min=-20.0, max=20.0)  # 更保守的范围

        # 应用 action mask 并计算平均值
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        
        return loss, kl.mean()

# --- 新增：修改Replay Buffer的 Collate Function ---
# 这个函数现在需要处理包含多个键的字典，并对它们进行填充
def grpo_collate_fn(batch):
    collated = {}
    # 提取所有 experience 对象
    experiences = [exp for item in batch for exp in item]
    
    # 逐个处理 experience 的属性
    for key in experiences[0].__dict__.keys():
        # 获取每个 experience 的属性值
        values = [getattr(exp, key) for exp in experiences]
        
        if key == 'inputs': # 'inputs' 是一个字典
            # 单独处理 'inputs' 字典中的每个键
            input_keys = values[0].keys()
            collated_inputs = {}
            for input_key in input_keys:
                input_values = [v[input_key].squeeze(0) for v in values] # 移除批次维度
                if input_key in ['input_ids', 'attention_mask']:
                    # 对序列进行填充
                    collated_inputs[input_key] = pad_sequence(input_values, batch_first=True, padding_value=0)
                elif input_key == 'pixel_values':
                    # pixel_values 可能有不同的图片数量，需要特殊处理
                    # 简单起见，我们假设每个样本的图片数量在 collate 前已经统一
                    # 或者，如果RL的batch size为1，则无需处理
                    collated_inputs[input_key] = torch.cat(input_values, dim=0)
                elif input_key == 'tgt_sizes':
                     #  tgt_sizes 需要根据填充后的长度进行更新
                    new_tgt_sizes = []
                    padded_len = collated_inputs['input_ids'].shape[1]
                    for v in input_values:
                        new_tgt_sizes.append([v.shape[0], padded_len])
                    collated_inputs[input_key] = torch.tensor(new_tgt_sizes, dtype=torch.long)
                else:
                    # 对于其他不需要填充的张量
                    collated_inputs[input_key] = torch.stack(input_values)
            collated['inputs'] = collated_inputs
        elif isinstance(values[0], torch.Tensor):
            # 对其他Tensor属性进行填充或堆叠
            if values[0].ndim > 0: # 如果是序列
                collated[key] = pad_sequence([v.squeeze(0) for v in values], batch_first=True, padding_value=0)
            else:
                collated[key] = torch.stack(values)
        else: # 对于非Tensor属性，直接收集
            collated[key] = values
            
    # 返回一个类似 Experience 的结构，但值是批处理过的
    # 这里我们只返回一个字典，因为 GRPOLoss 函数会解析它
    return collated


# ===================================================================
# 请将你的 main 函数和 Experience 类完全替换为以下代码
# ===================================================================

@dataclass
class Experience:
    """A single step of experience."""
    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    log_probs_ref: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.Tensor]
    action_mask: torch.Tensor
    pixel_values: torch.Tensor
    tgt_sizes: torch.Tensor
    image_bound: list
    kl: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "Experience":
        members = {}
        for field in fields(self):
            v = getattr(self, field.name)
            if isinstance(v, torch.Tensor):
                v = v.to(device=device)
            members[field.name] = v
        return Experience(**members)

def move_to_device(v, device, key=None):
    if key == "image_bound":
        return [[]]
    if isinstance(v, torch.Tensor):
        return v.to(device)
    elif isinstance(v, list):
        return [move_to_device(i, device) for i in v]
    elif isinstance(v, dict):
        return {kk: move_to_device(vv, device, kk) for kk, vv in v.items()}
    else:
        return v

def get_first_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            result = get_first_tensor(item)
            if isinstance(result, torch.Tensor):
                return result
    # 如果都不是，返回 None
    return None

def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    """从 logits 计算序列的对数概率"""
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)

def sequences_log_probs(
    model,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """计算序列的对数概率"""
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    
    # 使用 forward 方法获取 logits
    output = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = output["logits"]
    log_probs = sequence_log_probs_from_logits(
        logits=logits[:, :-1].to(torch.float32),
        output_ids=sequence_ids[:, 1:],
    )
    return log_probs

def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """计算优势函数"""
    return (returns - returns.mean()) / (returns.std() + eps)

def calculate_reward(answer, sample_label):
    """
    分别对label和predict的l,w,h排序，计算排序后对应位置的差值
    reward = 1 - 平均相对误差
    """
    def parse_and_sort(label_str):
        """解析并排序"""
        numbers = re.findall(r'\d+\.?\d*', label_str)
        if len(numbers) >= 3:
            return sorted([float(x) for x in numbers[:3]], reverse=True)  # 从大到小排序
        elif len(numbers) == 2:
            return sorted([float(x) for x in numbers[:2]], reverse=True)
        elif len(numbers) == 1:
            return [float(numbers[0])]
        return None
    
    def parse_answer_and_sort(answer_str):
        """解析answer并排序"""
        try:
            answer_dict = json.loads(answer_str)
            vals = []
            if 'length' in answer_dict:
                vals.append(float(answer_dict['length']))
            if 'width' in answer_dict:
                vals.append(float(answer_dict['width']))
            if 'height' in answer_dict:
                vals.append(float(answer_dict['height']))
            return sorted(vals, reverse=True) if vals else None
        except:
            return parse_and_sort(answer_str)
    
    # 解析并排序
    label_sorted = parse_and_sort(sample_label)
    answer_sorted = parse_answer_and_sort(answer)
    
    if not label_sorted or not answer_sorted:
        return 0.0
    
    # 计算排序后对应位置的相对误差
    min_len = min(len(label_sorted), len(answer_sorted))
    total_error = 0.0
    
    for i in range(min_len):
        label_val = label_sorted[i]
        answer_val = answer_sorted[i]
        if label_val > 0:
            rel_error = abs(answer_val - label_val) / label_val
            total_error += rel_error
    
    avg_error = total_error / min_len
    reward = max(0.0, 1.0 - avg_error)
    
    return reward

def main():
    # 参数
    dir_quickllm = "/data2/users/yszhang/quickllm"
    model_path = f"{dir_quickllm}/models/OpenBMB/MiniCPM-o-2_6"
    data_path = f"{dir_quickllm}/learnings/dim_grpo/data.xlsx"
    log_dir = f"{dir_quickllm}/learnings/dim_grpo/runs/dim_grpo_minicpm_final_correct"
    image_cache_dir = f"{dir_quickllm}/learnings/dim_grpo/image_cache_new"
    output_dir = f"{dir_quickllm}/outputs/dim_grpo_minicpm_final_correct/"
    
    batch_size = 1
    grad_accumulation_steps = 4  # 增加回4，获得更多样本
    group_size = 2  # 保持2
    lr = 1e-6  # 降低学习率，提高训练稳定性
    clip_eps = 0.3  # 增加clip范围，允许更多策略探索
    kl_weight = 0.005  # 降低KL权重，减少对策略更新的约束
    max_length = 64  # 进一步减少到64
    checkpoint_interval = 50  # 增加保存间隔，减少磁盘占用
    max_steps = 100  # 设置最大训练步数

    # 添加generation_config定义
    generation_config = GenerationConfig(
        max_new_tokens=max_length,
        temperature=0.7,
        do_sample=True,
        pad_token_id=0,
        eos_token_id=2
    )

    # 内存优化设置
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 设置较小的max_length以减少内存使用
    max_length = 64  # 从128减少到64

    os.makedirs(output_dir, exist_ok=True)
    logger.add(os.path.join(output_dir, "train.log"), rotation="10 MB")
    logger.info(f"启动真实训练，输出目录: {output_dir}")

    # --- 接收所有三个组件 ---
    model, tokenizer, image_processor = load_minicpm(model_path)
    reference_model, _, _ = load_minicpm(model_path) # 参考模型不需要分词器和图像处理器
    reference_model.eval()
    
    # 设置模型为训练模式，但使用内存优化
    model.train()
    
    # 清理内存
    torch.cuda.empty_cache()

    dataset = DimDataset(data_path, image_cache_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    logger.info(f"数据加载完成，batch_size={batch_size}，总batch数={len(dataloader)}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)
    writer = SummaryWriter(log_dir)

    global_step = 0
    for epoch in range(10):  # 增加epoch数，但用步数控制
        logger.info(f"--- 开始 Epoch {epoch+1} ---")
        replay_buffer = ReplayBuffer()
        
        # 每个epoch开始时清理内存
        torch.cuda.empty_cache()

        for batch_idx, batch in enumerate(dataloader):
            if batch is None: continue
            
            # 检查是否达到最大步数
            if global_step >= max_steps:
                logger.info(f"达到最大训练步数 {max_steps}，训练结束")
                break

            sample_images = batch["images"][0]
            sample_text = batch["text"][0]
            sample_label = batch["label"][0]
            
            if not sample_images:
                logger.warning(f"跳过无图片的样本: {sample_text}")
                continue

            logger.info(f"处理Batch {batch_idx}: text='{sample_text[:50]}...', label='{sample_label}'")
            
            # 每个batch开始时清理内存
            torch.cuda.empty_cache()

            prompt_str = build_prompt(sample_text, sample_images)
            user_msgs = [{"role": "user", "content": prompt_str, "images": sample_images}]
            
            for group_idx in range(group_size):
                with torch.no_grad():
                    answer = model.chat(
                        msgs=user_msgs, tokenizer=tokenizer,
                        max_new_tokens=max_length, temperature=0.7
                    )
                    
                    reward_val = calculate_reward(answer, sample_label)

                    logger.info(f"生成{group_idx+1}: {answer[:50]}... | Reward: {reward_val:.3f}")

                    # --- Input Preparation (Using the separate image_processor) ---
                    assistant_msgs = [{"role": "assistant", "content": answer}]
                    full_msgs = user_msgs + assistant_msgs

                    prompt_text_parts = []
                    all_images = []
                    for msg in full_msgs:
                        role, content = msg['role'], msg['content']
                        if role == 'user':
                            prompt_text_parts.append(f"<s><用户>{content}")
                            if 'images' in msg and msg['images']:
                                all_images.extend(msg['images'])
                                prompt_text_parts.append('\n' + '<image>' * len(msg['images']))
                        elif role == 'assistant':
                            prompt_text_parts.append(f"</s><AI>{content}</s>")
                    full_prompt_text = "".join(prompt_text_parts)
                    inputs = tokenizer(full_prompt_text, return_tensors="pt")

                    # 获取 patch_size
                    patch_size = getattr(model.config, "patch_size", 14)

                    if all_images:
                        # 处理每张图片
                        processed_images = []
                        for img in all_images:
                            img = img.resize((224, 224), Image.Resampling.LANCZOS)
                            img_array = np.array(img)
                            if len(img_array.shape) == 2:
                                img_array = np.stack([img_array] * 3, axis=-1)
                            elif img_array.shape[-1] == 4:
                                img_array = img_array[..., :3]
                            img_tensor = torch.from_numpy(img_array).float()
                            img_tensor = img_tensor.permute(2, 0, 1) / 255.0  # [3, 224, 224]
                            processed_images.append(img_tensor)
                        
                        if processed_images:
                            # 将所有图片堆叠成一个 batch
                            pixel_values = torch.stack(processed_images, dim=0)  # [N, 3, 224, 224]
                            # 将多张图片合并为一张
                            pixel_values = pixel_values.mean(dim=0)  # [3, 224, 224]
                            # 添加 batch 维度
                            pixel_values = pixel_values.unsqueeze(0)  # [1, 3, 224, 224]
                            # 将 pixel_values 转换为列表
                            pixel_values = [pixel_values]
                            # 生成对应的 tgt_sizes
                            tgt_sizes = torch.tensor([[224 // patch_size, 224 // patch_size]], dtype=torch.long)
                        else:
                            pixel_values = [torch.zeros((1, 3, 224, 224), dtype=torch.float32)]
                            tgt_sizes = torch.tensor([[224 // patch_size, 224 // patch_size]], dtype=torch.long)
                    else:
                        pixel_values = [torch.zeros((1, 3, 224, 224), dtype=torch.float32)]
                        tgt_sizes = torch.tensor([[224 // patch_size, 224 // patch_size]], dtype=torch.long)

                    seq_len = inputs.input_ids.shape[1]
                    data = {
                        "input_ids": inputs.input_ids,
                        "pixel_values": pixel_values,
                        "attention_mask": inputs.attention_mask,
                        "tgt_sizes": tgt_sizes,
                        "image_bound": [[] for _ in range(inputs.input_ids.shape[0])]
                    }

                    action_mask = torch.zeros(seq_len, dtype=torch.bool)
                    action_mask[seq_len - 1] = True

                    # 自动查找 <image> token 在 input_ids 里的位置，生成 image_bound
                    image_token = '<image>'
                    image_token_id = tokenizer.convert_tokens_to_ids(image_token)
                    input_ids_tensor = inputs.input_ids[0]  # 假设 batch=1
                    image_token_positions = (input_ids_tensor == image_token_id).nonzero(as_tuple=True)[0].tolist()
                    image_bound = [[ [pos, pos+1] for pos in image_token_positions ]]
                    data_on_device = {k: move_to_device(v, model.device, k) for k, v in data.items() if v is not None}
                    data_on_device['image_bound'] = image_bound

                    # 先生成outputs
                    behavior_outputs = model.generate(
                        input_ids=data_on_device["input_ids"],
                        pixel_values=data_on_device["pixel_values"],
                        tgt_sizes=data_on_device["tgt_sizes"],
                        attention_mask=data_on_device["attention_mask"],
                        image_bound=data_on_device["image_bound"],
                        tokenizer=tokenizer,
                        generation_config=generation_config
                    )

                    # 再处理outputs
                    if behavior_outputs[1].scores is not None and len(behavior_outputs[1].scores) > 0:
                        behavior_logits = behavior_outputs[1].scores[-1].float()
                    else:
                        behavior_logits = behavior_outputs[1].sequences.float()

                    behavior_log_probs = F.log_softmax(behavior_logits, dim=-1)
                    last_token_id = data_on_device["input_ids"][0, -1]
                    
                    # 确保last_token_id在有效范围内
                    if last_token_id >= behavior_logits.shape[1]:
                        logger.warning(f"last_token_id {last_token_id} 超出范围 {behavior_logits.shape[1]}，使用0")
                        last_token_id = torch.tensor(0, device=last_token_id.device)
                    
                    behavior_action_log_probs = behavior_log_probs.gather(dim=-1, index=last_token_id.unsqueeze(0).unsqueeze(-1)).squeeze(-1)

                    # 参考模型也使用 generate 方法
                    ref_outputs = reference_model.generate(
                        input_ids=data_on_device["input_ids"],
                        pixel_values=data_on_device["pixel_values"],
                        tgt_sizes=data_on_device["tgt_sizes"],
                        attention_mask=data_on_device["attention_mask"],
                        image_bound=data_on_device["image_bound"],
                        tokenizer=tokenizer,
                        generation_config=generation_config
                    )
                    
                    # 使用 forward 方法获取参考模型的 logits，确保计算正确
                    with torch.no_grad():
                        ref_data = {
                            "input_ids": data_on_device["input_ids"],
                            "attention_mask": data_on_device["attention_mask"],
                            "pixel_values": data_on_device["pixel_values"],
                            "tgt_sizes": data_on_device["tgt_sizes"],
                            "image_bound": data_on_device["image_bound"],
                            "position_ids": torch.arange(data_on_device["input_ids"].shape[1], dtype=torch.long, device=reference_model.device).unsqueeze(0)
                        }
                        
                        ref_forward_output = reference_model(data=ref_data, use_cache=False)
                        
                        # 获取最后一个位置的 logits
                        last_token_pos = data_on_device["input_ids"].shape[1] - 1
                        ref_last_logits = ref_forward_output.logits[:, last_token_pos, :].float()
                        
                        # 获取最后一个token的ID
                        ref_last_token_id = data_on_device["input_ids"][0, last_token_pos]
                        
                        # 确保last_token_id在有效范围内
                        if ref_last_token_id >= ref_last_logits.shape[1]:
                            logger.warning(f"ref_last_logits中last_token_id {ref_last_token_id} 超出范围 {ref_last_logits.shape[1]}，使用0")
                            ref_last_token_id = torch.tensor(0, device=ref_last_token_id.device)
                        
                        # 计算参考模型最后一个token的对数概率
                        ref_log_probs = F.log_softmax(ref_last_logits, dim=-1)
                        ref_action_log_probs = ref_log_probs.gather(dim=-1, index=ref_last_token_id.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
                    
                    # 创建 Experience 对象
                    exp = Experience(
                        sequences=data_on_device["input_ids"],
                        action_log_probs=behavior_action_log_probs,
                        log_probs_ref=ref_action_log_probs,
                        returns=torch.tensor([reward_val], dtype=torch.float32),
                        advantages=torch.tensor([0.0]),
                        attention_mask=data_on_device["attention_mask"],
                        action_mask=action_mask.unsqueeze(0),
                        pixel_values=data_on_device["pixel_values"][0],
                        tgt_sizes=data_on_device["tgt_sizes"],
                        image_bound=data_on_device["image_bound"],
                    )
                    
                    replay_buffer.append(exp)

            logger.info(f"Replay buffer 大小: {len(replay_buffer.items)}/{grad_accumulation_steps}")

            # --- Optimization Phase ---
            if len(replay_buffer.items) >= grad_accumulation_steps:
                optimizer.zero_grad()
                experience_batch = replay_buffer.sample(grad_accumulation_steps)
                all_returns = torch.stack([exp.returns for exp in experience_batch])
                mean_returns = all_returns.mean()
                
                # 修复advantage计算，使用更合理的方法
                if len(all_returns) == 1:
                    # 单个样本时，使用reward本身作为advantage
                    advantage = all_returns[0] - 0.5  # 假设平均reward为0.5
                else:
                    # 使用更稳定的advantage计算
                    mean_returns = all_returns.mean()
                    std_returns = all_returns.std() + 1e-8
                    advantage = (all_returns - mean_returns) / std_returns
                    
                    # 限制advantage的范围，避免极端值
                    advantage = torch.clamp(advantage, min=-3.0, max=3.0)
                
                total_loss = 0.0
                all_kl_divs = []  # 收集所有经验的KL散度
                for i, exp in enumerate(experience_batch):
                    # 使用对应样本的advantage
                    sample_advantage = advantage[i]
                    
                    # 使用经验中的所有字段
                    if isinstance(exp.sequences, torch.Tensor):
                        sequences_tensor = exp.sequences
                    elif isinstance(exp.sequences, (list, tuple)) and len(exp.sequences) > 0:
                        sequences_tensor = exp.sequences[0]
                    else:
                        # 如果无法获取序列，跳过这个经验
                        logger.warning(f"无法获取序列数据，跳过经验: {type(exp.sequences)}")
                        continue
                    
                    # 获取序列长度
                    if isinstance(sequences_tensor, torch.Tensor):
                        if len(sequences_tensor.shape) == 1:
                            # 如果是一维张量，直接使用长度
                            seq_len = sequences_tensor.shape[0]
                        else:
                            # 如果是二维张量，使用第二维
                            seq_len = sequences_tensor.shape[1]
                    else:
                        # 如果是tuple或其他格式，取第一个元素
                        if isinstance(sequences_tensor, (list, tuple)) and len(sequences_tensor) > 0:
                            first_seq = sequences_tensor[0]
                            if hasattr(first_seq, 'shape'):
                                seq_len = first_seq.shape[1] if len(first_seq.shape) > 1 else first_seq.shape[0]
                            else:
                                seq_len = len(first_seq)
                        else:
                            # 如果无法确定长度，使用一个默认值
                            seq_len = 256
                            logger.warning(f"无法确定序列长度，使用默认值: {seq_len}")
                    
                    # 确保sequences_tensor是张量
                    if not isinstance(sequences_tensor, torch.Tensor):
                        logger.warning(f"sequences_tensor不是张量，跳过经验: {type(sequences_tensor)}")
                        continue
                    
                    # 如果sequences_tensor是一维的，需要添加batch维度
                    if len(sequences_tensor.shape) == 1:
                        sequences_tensor = sequences_tensor.unsqueeze(0)  # 添加batch维度
                    
                    data = {
                        "input_ids": sequences_tensor.to(model.device),  # 已经添加了batch维度
                        "attention_mask": exp.attention_mask.to(model.device).unsqueeze(0) if exp.attention_mask is not None else None,
                        "pixel_values": [[exp.pixel_values.to(model.device)]],  # list of lists格式
                        "tgt_sizes": exp.tgt_sizes.to(model.device).unsqueeze(0),
                        "image_bound": [[[int(pos), int(pos+1)] for pos in exp.image_bound[0]]],
                        "position_ids": torch.arange(seq_len, dtype=torch.long, device=model.device).unsqueeze(0)
                    }
                    
                    # 确保 input_ids 和 pixel_values 的 batch 维度一致
                    assert data["input_ids"].size(0) == len(data["pixel_values"]), \
                        f"input_ids batch: {data['input_ids'].size(0)}, pixel_values batch: {len(data['pixel_values'])}"
                    
                    # 使用 forward 方法重新计算当前 log_probs，确保有真正的梯度
                    with torch.enable_grad():
                        last_token_pos = seq_len - 1
                        
                        # 使用 forward 方法获取 logits
                        forward_output = model(data=data, use_cache=False)
                        
                        # 获取最后一个位置的 logits
                        last_logits = forward_output.logits[:, last_token_pos, :].float()
                        
                        # 获取最后一个token的ID
                        last_token_id = sequences_tensor[0, last_token_pos] if len(sequences_tensor.shape) > 1 else sequences_tensor[last_token_pos]
                        
                        # 确保last_token_id在有效范围内
                        if last_token_id >= last_logits.shape[1]:
                            logger.warning(f"forward中last_token_id {last_token_id} 超出范围 {last_logits.shape[1]}，使用0")
                            last_token_id = torch.tensor(0, device=last_token_id.device)
                        
                        # 计算最后一个token的对数概率
                        log_probs = F.log_softmax(last_logits, dim=-1)
                        current_action_log_probs = log_probs.gather(dim=-1, index=last_token_id.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
                    
                    # 确保action_mask的维度正确
                    if len(exp.action_mask.shape) == 1:
                        # 如果action_mask是一维的，只取最后一个位置
                        action_mask = exp.action_mask[-1:].to(model.device)
                    else:
                        action_mask = exp.action_mask.to(model.device)
                    
                    loss, kl_div = objective(
                        log_probs=current_action_log_probs,
                        log_probs_old=exp.action_log_probs.to(model.device),
                        log_probs_ref=exp.log_probs_ref.to(model.device),
                        advantages=sample_advantage.to(model.device),
                        action_mask=action_mask
                    )
                    
                    # 收集KL散度值，但处理异常大的值
                    kl_value = kl_div.item()
                    
                    if kl_value > 1000:
                        logger.warning(f"跳过异常大的KL值: {kl_value:.6f}")
                        kl_value = 1000.0  # 限制在合理范围内用于训练
                    
                    all_kl_divs.append(kl_value)
                    
                    loss = loss / grad_accumulation_steps
                    
                    # 检查loss是否为有效值，避免inf导致的backward问题
                    if loss.requires_grad and not torch.isnan(loss) and not torch.isinf(loss):
                        # 清理内存
                        torch.cuda.empty_cache()
                        
                        # 使用try-except包装backward，避免内存不足
                        try:
                            loss.backward()
                            total_loss += loss.item()
                        except torch.cuda.OutOfMemoryError:
                            logger.warning("内存不足，跳过此步的backward")
                            torch.cuda.empty_cache()
                            total_loss += 0.0
                    else:
                        logger.warning(f"跳过无效的loss: {loss.item()}")
                        total_loss += 0.0
                    
                    # 每个经验处理后清理内存
                    torch.cuda.empty_cache()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # 清理内存
                torch.cuda.empty_cache()
                
                # 使用try-except包装optimizer.step()
                try:
                    optimizer.step()
                except torch.cuda.OutOfMemoryError:
                    logger.warning("optimizer.step()内存不足，跳过此步更新")
                    torch.cuda.empty_cache()
                    # 重置梯度
                    optimizer.zero_grad()
                
                # 计算平均KL散度
                mean_kl_div = sum(all_kl_divs) / len(all_kl_divs) if all_kl_divs else 0.0
                
                # 计算平均advantage
                mean_advantage = advantage.mean().item() if len(advantage) > 0 else 0.0
                
                logger.success(f"Step={global_step}/{max_steps}, Loss={total_loss * grad_accumulation_steps:.4f}, KL={mean_kl_div:.4f}, Reward={all_returns.mean().item():.3f}, Advantage={mean_advantage:.3f}")
                writer.add_scalar("train/loss", total_loss * grad_accumulation_steps, global_step)
                writer.add_scalar("train/kl", mean_kl_div, global_step)
                writer.add_scalar("train/reward", all_returns.mean().item(), global_step)
                writer.add_scalar("train/advantage", mean_advantage, global_step)
                
                # 添加训练进度信息
                progress = global_step / max_steps * 100
                logger.info(f"训练进度: {progress:.1f}% ({global_step}/{max_steps})")
                
                global_step += 1
                replay_buffer.clear()

                # 修改保存策略：只保存两次结果
                if global_step == 25:  # 第一次保存：训练1/4时
                    save_path = os.path.join(output_dir, f"model_step{global_step}.pt")
                    torch.save(model.state_dict(), save_path)
                    logger.info(f"保存模型到: {save_path}")
                elif global_step == max_steps:  # 第二次保存：训练完成时
                    save_path = os.path.join(output_dir, f"model_step{global_step}.pt")
                    torch.save(model.state_dict(), save_path)
                    logger.info(f"保存最终模型到: {save_path}")
                
                # 添加训练步数限制
                if global_step >= max_steps:
                    logger.info(f"达到最大训练步数 {max_steps}，训练结束")
                    break
                
                logger.info("=" * 80)  # 添加分隔符

    writer.close()
    logger.info("训练结束")


if __name__ == "__main__":
    main()