# -*- coding: utf-8 -*-
"""
Step 8: 模型训练过程
实现完整的训练循环，包括损失函数、训练步骤、验证评估等
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from transformers import get_cosine_schedule_with_warmup
from datetime import datetime
from loguru import logger
import matplotlib.pyplot as plt

# 设置GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 导入必要的组件（避免循环导入，直接包含函数）
import math

# 必要的辅助函数
def create_padding_mask(batch_data: torch.Tensor, pad_token_id: int = 0):
    """创建padding mask"""
    mask = (batch_data == pad_token_id).float()
    return mask[:, None, None, :]  # [B, 1, 1, L]

def create_look_ahead_mask(size: int):
    """生成Look-ahead mask (上三角矩阵)"""
    ones = torch.ones((size, size))
    mask = torch.triu(ones, diagonal=1)
    return mask

def create_masks(
        inp_ids: torch.Tensor,  # [B, L_src]
        tar_ids: torch.Tensor,  # [B, L_tgt] —— 通常是 decoder 输入（已左移）
        src_pad_id: int = 0,
        tgt_pad_id: int = 0,
):
    """
    创建各种mask
    返回:
      encoder_padding_mask         : [B, 1, 1, L_src]  (给 EncoderLayer self-attn)
      decoder_mask (LA + padding)  : [B, 1, L_tgt, L_tgt]  (给 DecoderLayer 自注意力)
      encoder_decoder_padding_mask : [B, 1, 1, L_src]  (给 DecoderLayer cross-attn)
    语义:
      1 = 屏蔽（masked），0 = 保留
    """
    # 1) Encoder 端 padding mask
    encoder_padding_mask = create_padding_mask(inp_ids, pad_token_id=src_pad_id)  # [B,1,1,L_src]
    encoder_decoder_padding_mask = create_padding_mask(inp_ids, pad_token_id=src_pad_id)  # [B,1,1,L_src]

    # 2) Decoder 端 look-ahead + padding 合并
    B, L_tgt = tar_ids.size(0), tar_ids.size(1)

    # [L_tgt, L_tgt] → [1,1,L_tgt,L_tgt]，放到与输入相同 device/dtype
    look_ahead = create_look_ahead_mask(L_tgt).to(
        device=tar_ids.device, dtype=encoder_padding_mask.dtype
    ).unsqueeze(0).unsqueeze(1)  # [1,1,L_tgt,L_tgt]

    # 目标端 padding： [B,1,1,L_tgt] → 扩到 [B,1,L_tgt,L_tgt]
    decoder_padding_mask = create_padding_mask(tar_ids, pad_token_id=tgt_pad_id)  # [B,1,1,L_tgt]
    decoder_padding_mask = decoder_padding_mask.expand(-1, -1, L_tgt, -1)  # [B,1,L_tgt,L_tgt]

    # 合并（任一为 1 即屏蔽）
    decoder_mask = torch.maximum(decoder_padding_mask, look_ahead)  # [B,1,L_tgt,L_tgt]

    return encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask

# 简单的Transformer模拟类（用于演示）
class Transformer(nn.Module):
    """简单的Transformer模拟类"""
    def __init__(self, num_layers, input_vocab_size, target_vocab_size,
                 max_length, d_model, num_heads, dff, rate=0.1,
                 src_padding_idx: int = None, tgt_padding_idx: int = None):
        super().__init__()
        self.encoder_embedding = nn.Embedding(input_vocab_size, d_model, padding_idx=src_padding_idx)
        self.decoder_embedding = nn.Embedding(target_vocab_size, d_model, padding_idx=tgt_padding_idx)
        self.final_layer = nn.Linear(d_model, target_vocab_size)
        
    def forward(self, inp_ids, tgt_ids, src_mask=None, tgt_mask=None, enc_dec_mask=None):
        # 简单的模拟前向传播
        enc_out = self.encoder_embedding(inp_ids)
        dec_out = self.decoder_embedding(tgt_ids)
        logits = self.final_layer(dec_out)
        return logits, {}

class AverageMeter:
    """用于计算平均值的工具类"""
    def __init__(self, name="meter"): 
        self.name = name
        self.reset()

    def reset(self): 
        self.sum = 0.0
        self.n = 0

    def update(self, val, count=1): 
        self.sum += float(val) * count
        self.n += count

    @property
    def avg(self): 
        return self.sum / max(1, self.n)

@torch.no_grad()
def token_accuracy(real, pred, pad_id):
    """计算token级别的准确率"""
    pred_ids = pred.argmax(dim=-1)  # (B, L)
    mask = (real != pad_id)
    correct = ((pred_ids == real) & mask).sum().item()
    denom = mask.sum().item()
    return correct / max(1, denom)

def loss_function(real, pred, loss_object):
    """
    计算损失函数
    Args:
        real: (B, L) target ids (shift 后)
        pred: (B, L, V) logits
        loss_object: 损失函数对象
    Returns:
        loss (float): 平均有效 token 的交叉熵损失
    """
    B, L, V = pred.shape

    # 展平
    pred = pred.reshape(-1, V)  # (B*L, V)
    real = real.reshape(-1)  # (B*L,)

    # token 级别交叉熵 (padding 已被 ignore_index 屏蔽)
    loss_ = loss_object(pred, real)  # (B*L,)

    return loss_.mean()

def train_step(batch, transformer, optimizer, scheduler=None, device=None, 
               pt_tokenizer=None, en_tokenizer=None, loss_object=None):
    """执行一个训练步骤"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    transformer.train()

    inp = batch["pt_input_ids"].to(device)
    tar = batch["en_input_ids"].to(device)

    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    SRC_PAD_ID = pt_tokenizer.pad_token_id
    TGT_PAD_ID = en_tokenizer.pad_token_id

    enc_pad_mask, dec_mask, enc_dec_pad_mask = create_masks(
        inp, tar_inp, src_pad_id=SRC_PAD_ID, tgt_pad_id=TGT_PAD_ID
    )
    enc_dec_mask = enc_dec_pad_mask.expand(-1, 1, tar_inp.size(1), -1)

    logits, _ = transformer(
        inp, tar_inp,
        src_mask=enc_pad_mask,
        tgt_mask=dec_mask,
        enc_dec_mask=enc_dec_mask
    )

    loss = loss_function(tar_real, logits, loss_object)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    acc = token_accuracy(tar_real, logits, pad_id=TGT_PAD_ID)
    return loss.item(), acc

@torch.no_grad()
def evaluate_on_val(model, val_loader, device, pt_tokenizer, en_tokenizer, loss_object):
    """在验证集上评估模型"""
    model.eval()
    total_loss = 0
    total_acc = 0
    total_count = 0

    for batch in val_loader:
        inp = batch["pt_input_ids"].to(device)
        tar = batch["en_input_ids"].to(device)

        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_pad_mask, dec_mask, enc_dec_pad_mask = create_masks(
            inp, tar_inp, src_pad_id=pt_tokenizer.pad_token_id, tgt_pad_id=en_tokenizer.pad_token_id
        )
        enc_dec_mask = enc_dec_pad_mask.expand(-1, 1, tar_inp.size(1), -1)

        logits, _ = model(
            inp, tar_inp,
            src_mask=enc_pad_mask,
            tgt_mask=dec_mask,
            enc_dec_mask=enc_dec_mask
        )

        loss = loss_function(tar_real, logits, loss_object)
        acc = token_accuracy(tar_real, logits, pad_id=en_tokenizer.pad_token_id)

        total_loss += loss.item() * inp.size(0)
        total_acc += acc * inp.size(0)
        total_count += inp.size(0)

    avg_loss = total_loss / total_count
    avg_acc = total_acc / total_count
    return avg_loss, avg_acc

def save_ckpt(model, optimizer, scheduler, epoch, step, ckpt_dir="checkpoints", tag="latest"):
    """保存checkpoint"""
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "sched": scheduler.state_dict() if scheduler else None,
    }

    latest_path = os.path.join(ckpt_dir, "latest.pt")
    torch.save(ckpt, latest_path)

    if tag == "latest":
        path = os.path.join(ckpt_dir, f"mid_e{epoch}_s{step}.pt")
    elif tag == "error":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(ckpt_dir, f"error_e{epoch}_s{step}_{ts}.pt")
    else:
        path = os.path.join(ckpt_dir, f"{tag}_e{epoch}_s{step}.pt")

    torch.save(ckpt, path)
    return path

def load_ckpt(model, optimizer=None, scheduler=None, ckpt_dir="checkpoints", device="cpu"):
    """加载checkpoint"""
    latest = os.path.join(ckpt_dir, "latest.pt")
    if not os.path.exists(latest):
        logger.info("⚠️ No checkpoint found, training from scratch.")
        return 0, 0
    ckpt = torch.load(latest, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer: 
        optimizer.load_state_dict(ckpt["optim"])
    if scheduler and ckpt["sched"]: 
        scheduler.load_state_dict(ckpt["sched"])
    logger.info(f"✅ checkpoint loaded (epoch={ckpt['epoch']}, step={ckpt['step']})")
    return ckpt["epoch"], ckpt["step"]

def train_model(
        epochs: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        scheduler=None,
        device: str = None,
        log_every: int = 100,
        ckpt_dir: str = "checkpoints",
        ckpt_prefix: str = "ckpt",
        pt_tokenizer=None,
        en_tokenizer=None,
        loss_object=None,
):
    """训练模型的主函数"""
    os.makedirs(ckpt_dir, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_loss_meter = AverageMeter("train_loss")
    train_acc_meter = AverageMeter("train_accuracy")
    global_step = 0

    # 记录训练历史
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        try:
            start = time.time()
            train_loss_meter.reset()
            train_acc_meter.reset()
            model.train()

            for batch_idx, batch in enumerate(train_loader):
                loss_val, acc_val = train_step(
                    batch=batch, 
                    transformer=model, 
                    optimizer=optimizer, 
                    scheduler=scheduler, 
                    device=device,
                    pt_tokenizer=pt_tokenizer,
                    en_tokenizer=en_tokenizer,
                    loss_object=loss_object
                )
                train_loss_meter.update(loss_val, 1)
                train_acc_meter.update(acc_val, 1)

                global_step += 1
                if batch_idx % log_every == 0:
                    logger.info(
                        f"Epoch {epoch + 1} Batch {batch_idx} global_step {global_step} "
                        f"Loss {train_loss_meter.avg:.4f} Accuracy {train_acc_meter.avg:.4f}"
                    )
                    save_ckpt(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch + 1,
                        step=global_step,
                        ckpt_dir=ckpt_dir,
                        tag="latest"
                    )

            logger.info(f"Epoch {epoch + 1} Loss {train_loss_meter.avg:.4f} Accuracy {train_acc_meter.avg:.4f}")
            logger.info(f"Time taken for 1 epoch: {time.time() - start:.2f} secs\n")

            # 记录训练指标
            train_losses.append(train_loss_meter.avg)
            train_accs.append(train_acc_meter.avg)

            # 每个epoch结束后进行验证集评测
            validate_loss, validate_acc = evaluate_on_val(
                model, val_loader, device, pt_tokenizer, en_tokenizer, loss_object
            )
            logger.info(f"Validation - Epoch {epoch + 1} Loss: {validate_loss:.4f}, Accuracy: {validate_acc:.4f}\n")
            
            # 记录验证指标
            val_losses.append(validate_loss)
            val_accs.append(validate_acc)

        except Exception as e:
            logger.info(f"报错啦!!! 报错信息: {e}")
            save_ckpt(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=global_step,
                tag="error"
            )
            break

    # 绘制训练曲线
    plot_training_curves(train_losses, train_accs, val_losses, val_accs)
    
    return train_losses, train_accs, val_losses, val_accs

def plot_training_curves(train_losses, train_accs, val_losses, val_accs):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(train_accs, label='Train Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=" * 60)
    print("Step 8: 模型训练过程")
    print("=" * 60)
    
    # 训练参数
    epochs = 10  # 为了演示，只训练10个epoch
    batch_size = 32
    learning_rate = 5e-4
    betas = (0.9, 0.98)
    eps = 1e-8
    weight_decay = 2e-5
    
    # 模型结构参数
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.25
    max_length = 30
    
    # 词表大小（示例值）
    input_vocab_size = 8192
    target_vocab_size = 8192
    
    print(f"🔧 训练参数:")
    print(f"   训练轮数: {epochs}")
    print(f"   批大小: {batch_size}")
    print(f"   学习率: {learning_rate}")
    print(f"   模型维度: {d_model}")
    print(f"   最大序列长度: {max_length}")
    
    try:
        # 1. 创建模型
        print(f"\n🔨 创建Transformer模型...")
        model = Transformer(
            num_layers=num_layers,
            input_vocab_size=input_vocab_size,
            target_vocab_size=target_vocab_size,
            max_length=max_length,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            rate=dropout_rate,
            src_padding_idx=0,
            tgt_padding_idx=0,
        )
        
        # 2. 创建优化器和调度器
        print(f"\n🔨 创建优化器和调度器...")
        train_loader_length = 100  # 假设训练集有100个batch
        num_training_steps = train_loader_length * epochs
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        
        warmup_steps = int(0.1 * num_training_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=0.5,
        )
        
        # 3. 创建损失函数
        print(f"\n🔨 创建损失函数...")
        PAD_ID_TGT = 0  # 假设pad_token_id=0
        loss_object = nn.CrossEntropyLoss(reduction="none", ignore_index=PAD_ID_TGT)
        
        # 4. 创建模拟的tokenizer（用于演示）
        print(f"\n🔨 创建模拟tokenizer...")
        class MockTokenizer:
            def __init__(self, vocab_size, pad_token_id=0):
                self.vocab_size = vocab_size
                self.pad_token_id = pad_token_id
        
        pt_tokenizer = MockTokenizer(input_vocab_size, 0)
        en_tokenizer = MockTokenizer(target_vocab_size, 0)
        
        # 5. 创建模拟的DataLoader（用于演示）
        print(f"\n🔨 创建模拟DataLoader...")
        class MockDataLoader:
            def __init__(self, batch_size, num_batches):
                self.batch_size = batch_size
                self.num_batches = num_batches
            
            def __iter__(self):
                for _ in range(self.num_batches):
                    # 创建模拟的batch数据
                    batch = {
                        "pt_input_ids": torch.randint(0, input_vocab_size, (batch_size, max_length//2)),
                        "en_input_ids": torch.randint(0, target_vocab_size, (batch_size, max_length//2)),
                        "pt_attention_mask": torch.ones(batch_size, max_length//2),
                        "en_attention_mask": torch.ones(batch_size, max_length//2),
                    }
                    yield batch
            
            def __len__(self):
                return self.num_batches
        
        train_loader = MockDataLoader(batch_size, train_loader_length)
        val_loader = MockDataLoader(batch_size, 20)  # 验证集20个batch
        
        # 6. 开始训练
        print(f"\n🚀 开始训练...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   使用设备: {device}")
        
        train_losses, train_accs, val_losses, val_accs = train_model(
            epochs=epochs,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            scheduler=scheduler,
            device=device,
            log_every=20,  # 每20个batch打印一次
            ckpt_dir="checkpoints",
            ckpt_prefix="transformer",
            pt_tokenizer=pt_tokenizer,
            en_tokenizer=en_tokenizer,
            loss_object=loss_object,
        )
        
        print(f"\n✅ 训练完成！")
        print(f"📊 最终训练损失: {train_losses[-1]:.4f}")
        print(f"📊 最终训练准确率: {train_accs[-1]:.4f}")
        print(f"📊 最终验证损失: {val_losses[-1]:.4f}")
        print(f"📊 最终验证准确率: {val_accs[-1]:.4f}")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        print("💡 请检查参数设置是否正确")
