from collections.abc import Callable
import json
from pathlib import Path
import random
import re
from typing import Any, Iterator, Optional
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
import pandas as pd
from loss import approx_kl_divergence, GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch


def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    bf16: bool = True,
    device_map=None,
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
    )
    return model, tokenizer


# DeepSeek Zero system prompt
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""


@torch.no_grad()
def rollout(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    task: str,
    oracle_answer: str,
    num_rollouts: int,
    max_length: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:

    model.eval()

    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": task,
        },
    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to(device)

    # duplicate prompt num_rollouts times
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        num_rollouts, 1
    )

    input_ids = model_inputs["input_ids"].repeat(num_rollouts, 1)
    model_inputs["input_ids"] = input_ids

    # 2. sample completions
    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
    completions = tokenizer.batch_decode(
        sequence_ids[:, input_ids.shape[1] :], skip_special_tokens=True
    )

    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, input_ids.shape[1] :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]

    # 3. determine rewards
    returns = torch.zeros(num_rollouts, 1, dtype=torch.float)
    for i, completion in enumerate(completions):
        # search answer tag
        answer_match = re.search(
            r"<answer>(.*?)</answer>",
            completion,
            flags=re.DOTALL,
        )

        answer = answer_match.group(1) if answer_match else None
        reward = 0
        if answer is not None:
            if answer == oracle_answer:
                reward = 1.0
            elif oracle_answer in answer:
                reward = 0.5
            else:
                reward = 0.01

        returns[i] = reward

    return sequence_ids, returns.to(sequence_ids.device), action_mask, completions


def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)


def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std() + eps)


def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def sequences_log_probs(
    model: AutoModelForCausalLM,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
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


def read_jsonl(file_name: str | Path) -> Iterator:
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_prompts(
    file_name: str,
    predicate: Optional[Callable[[Any], bool]] = None,
    max_rows: Optional[int] = None,
) -> list:
    rows = []
    for x in read_jsonl(file_name):
        if predicate is None or predicate(x):
            rows.append(x)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def read_prompts_from_parquet(
    file_path: str | Path,
    max_rows: Optional[int] = None,
) -> list:
    """从parquet文件读取数据"""
    df = pd.read_parquet(file_path)
    
    # 转换为list of dict格式，用于兼容原有的数据处理逻辑
    rows = []
    for _, row in df.iterrows():
        # 使用英文原问题和详细答案，保持完整的推理链
        # 从详细答案中提取最终数字答案用于奖励计算
        answer_text = row['answer']
        final_answer = str(row['answer_only'])  # 提取最终数字答案用于奖励匹配
        
        data_item = {
            'question': row['question'],  # 使用英文原问题
            'answer': final_answer,  # 使用数字答案用于奖励匹配
            'full_answer': answer_text,  # 保存完整答案用于可能的参考
            'id': str(len(rows)),  # 生成ID
        }
        rows.append(data_item)
    
    return rows


def main():
    seed = 42
    wandb_project = None  # "tiny_grpo"
    device_index = 3
    model_name = "/data2/users/yszhang/quickllm/models/Qwen2.5-0.5B-Instruct"
    checkpoint_path = Path("/data2/users/yszhang/quickllm/outputs/tiny_grpo/output")
    checkpoint_interval = 20
    train_batch_size = 16
    lr = 5e-6
    kl_weight = 0.01
    clip_eps = 0.2

    group_size = 12
    rollouts_per_step = 32
    epochs_per_step = 1
    max_norm = 1.0  # gradient clipping

    # rollout params
    max_length = 1024
    top_p = 1.0
    temperature = 1.0

    device = torch.device("cuda", device_index)
    cpu_device = torch.device("cpu")
    init_rng(seed)

    reference_model, _ = load_model(model_name, device_map=device)
    model, tokenizer = load_model(model_name, device_map=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    reference_model.eval()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    pad_token_id = tokenizer.eos_token_id

    # 使用GSM8K中文数据集
    gsm8k_path = "/data2/users/yszhang/quickllm/rl/llm_related-main/grpo_from_scratch/datasets/gsm8k_chinese/data/train-00000-of-00001.parquet"
    prompts = read_prompts_from_parquet(
        gsm8k_path,
    )
    print(f"found {len(prompts)} matching prompts")
    prompt_loader = DataLoader(
        prompts,
        batch_size=rollouts_per_step,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )

    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)

    if wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=wandb_project)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(f'./runs/grpo_train')
    
    # Initialize loss history tracking (essential metrics only)
    loss_history = {
        "total_loss": [],
        "kl_divergence": [],
        "grad_norm": [],
        "returns": [],
        "success_rate": []
    }
    
    total_steps = len(prompt_loader)
    print(f"Total training steps: {total_steps}")
    print(f"Each step processes {rollouts_per_step} questions with {group_size} rollouts each")
    print(f"Training batch size: {train_batch_size}, epochs per step: {epochs_per_step}")

    for k, prompt_batch in enumerate(prompt_loader):
        rollout_returns = []

        replay_buffer.clear()

        questions = prompt_batch["question"]
        answers = prompt_batch["answer"]
        full_answers = prompt_batch["full_answer"]

        with torch.no_grad():
            for q, a, full_a in zip(questions, answers, full_answers):
                sequence_ids, returns, action_mask, completions = rollout(
                    model,
                    tokenizer,
                    q,
                    a,
                    num_rollouts=group_size,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    device=device,
                )

                print(f"===:rollout q='{q}', a='{a}', returns={returns.sum().item():.2f}, replay_buffer_size={len(replay_buffer)}, sequence_ids={sequence_ids.shape}")
                
                # 对比标准答案和模型top1答案
                best_idx = returns.argmax().item()  # 找到奖励最高的答案
                print(f"===:标准答案: {full_a[:200]}..." if len(full_a) > 200 else f"标准答案: {full_a}")
                print(f"===:模型top1: {completions[best_idx][:200]}..." if len(completions[best_idx]) > 200 else f"模型top1: {completions[best_idx]}")
                print("--------------------------------")
                rollout_returns.append(returns.cpu())

                advantages = group_advantages(returns)
                attention_mask = sequence_ids != pad_token_id

                log_probs = sequences_log_probs(
                    model=model,
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask,
                )
                log_probs_ref = sequences_log_probs(
                    model=reference_model,
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask,
                )
                kl = approx_kl_divergence(
                    log_probs=log_probs,
                    log_probs_ref=log_probs_ref,
                    action_mask=action_mask,
                )

                experience = Experience(
                    sequences=sequence_ids,
                    action_log_probs=log_probs,
                    log_probs_ref=log_probs_ref,
                    returns=returns,
                    advantages=advantages,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    kl=kl,
                )
                replay_buffer.append(experience.to(cpu_device))

        torch.cuda.empty_cache()
        episode_return_sum = torch.stack(rollout_returns).sum()
        print(f"returns of step {k}: {episode_return_sum:.4f}")
        wandb.log({"returns": episode_return_sum})
        
        # Log essential rollout metrics to TensorBoard
        writer.add_scalar("rollout/returns", episode_return_sum, k)
        all_returns = torch.cat(rollout_returns, dim=0)
        success_rate = (all_returns > 0.9).float().mean()
        writer.add_scalar("rollout/success_rate", success_rate, k)
        
        # Store essential rollout metrics in history
        loss_history["returns"].append(episode_return_sum.item())
        loss_history["success_rate"].append(success_rate.item())

        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=join_experience_batch,
        )

        for step_epoch in range(epochs_per_step):
            model.train()

            for batch_idx, exp in enumerate(experience_sampler):
                exp: Experience

                exp = exp.to(device)

                optimizer.zero_grad()

                log_probs = sequences_log_probs(
                    model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
                )

                loss, kl = objective(log_probs=log_probs, experience=exp)

                if not loss.isfinite():
                    print(f"Loss not finite, skipping backward, loss={loss}")
                    print(f"experience.advantages={experience.advantages}")
                    continue



                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)
                
                # Simplified logging - only essential metrics
                print(f"{step_epoch}: loss={loss:.4f}, kl={kl:.4f}, grad_norm={grad_norm:.4f}")
                wandb.log({"loss": loss, "kl": kl, "grad_norm": grad_norm})
                
                # Log essential training metrics to TensorBoard
                global_step = k * epochs_per_step * len(experience_sampler) + step_epoch * len(experience_sampler) + batch_idx
                writer.add_scalar("train/loss", loss, global_step)
                writer.add_scalar("train/kl_divergence", kl, global_step)
                writer.add_scalar("train/grad_norm", grad_norm, global_step)
                
                # Store essential training metrics in history
                loss_history["total_loss"].append(loss.item())
                loss_history["kl_divergence"].append(kl.item())
                loss_history["grad_norm"].append(grad_norm.item())

                optimizer.step()

        if (
            checkpoint_path is not None
            and checkpoint_interval is not None
            and (k + 1) % checkpoint_interval == 0
        ):
            model.save_pretrained(checkpoint_path / f"step_{k}")

    if checkpoint_path is not None:
        model.save_pretrained(checkpoint_path / f"step_{k}")
    
    # Save loss history to JSON file
    import json
    history_path = checkpoint_path / "loss_history.json"
    with open(history_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    print(f"Loss history saved to: {history_path}")
    
    # Close TensorBoard writer
    writer.close()
    print("Training completed!")
    
    # Print final statistics
    if len(loss_history["total_loss"]) > 0:
        print("\n=== Final Training Statistics ===")
        print(f"Final loss: {loss_history['total_loss'][-1]:.4f}")
        print(f"Final success rate: {loss_history['success_rate'][-1]:.2%}")
        print(f"Average return: {sum(loss_history['returns'])/len(loss_history['returns']):.2f}")
        print("="*40)


if __name__ == "__main__":
    main()
