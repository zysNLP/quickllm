import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 定义路径和参数
initial_model_path = "/data2/users/yszhang/quickllm/models/Qwen2.5-0.5B-Instruct"
trained_model_path = "/data2/users/yszhang/quickllm/outputs/tiny_grpo/output/step_232"  # 请将 {k} 替换为实际的最终步数
dataset_path = "/data2/users/yszhang/quickllm/rl/llm_related-main/grpo_from_scratch/datasets/gsm8k_chinese/data/train-00000-of-00001.parquet"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 tokenizer（两模型共用）
tokenizer = AutoTokenizer.from_pretrained(initial_model_path)
tokenizer.pad_token = tokenizer.eos_token

# 加载模型的函数
def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()
    return model

# 加载初始和训练后的模型
initial_model = load_model(initial_model_path)
trained_model = load_model(trained_model_path)

# 从数据集加载第一条问题
df = pd.read_parquet(dataset_path)
first_question = df.iloc[0]['question']
oracle_answer = df.iloc[0]['answer']

# 定义系统提示（与训练时一致）
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""

# 使用聊天模板格式化输入
chat_messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": first_question},
]
chat_prompt = tokenizer.apply_chat_template(
    chat_messages, tokenize=False, add_generation_prompt=True
)

# 编码输入
inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)

# 生成配置（与训练时一致）
generation_config = {
    "max_length": 1024,
    "temperature": 1.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

# 从初始模型生成输出
with torch.no_grad():
    initial_output_ids = initial_model.generate(**inputs, **generation_config)
    initial_output = tokenizer.decode(initial_output_ids[0], skip_special_tokens=True)


# 从训练后模型生成输出
with torch.no_grad():
    trained_output_ids = trained_model.generate(**inputs, **generation_config)
    trained_output = tokenizer.decode(trained_output_ids[0], skip_special_tokens=True)

# 打印结果
print(f"问题: {first_question}")
print(f"标准答案: {oracle_answer}")
print("\n初始模型输出:")
print(initial_output)
print("\n训练后模型输出:")
print(trained_output)
