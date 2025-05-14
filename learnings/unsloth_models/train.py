#!/usr/bin/env python

# -- coding: utf-8 --
# 导入必要的库
import os # 导入 os 库用于处理文件路径
os.makedirs("/data2/datasets_cache", exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = "/data2/datasets_cache" # 设置数据集缓存目录到 /data2 分区
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
print(f"当前GPU设备：{os.environ['CUDA_VISIBLE_DEVICES']}")

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset  # 导入 datasets 库的 Dataset 类，用于从列表中创建数据集

# 1. 模型加载参数设置
max_seq_length = 2048
model_name = "/quickllm/us_models/r1"
load_in_4bit = True

# 加载预训练模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=load_in_4bit,
    token=None,
    device_map={'':torch.cuda.current_device()}
)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备：{device}")

# 2. 数据准备与格式化

train_prompt_style = (
    "你是一位法律专家，具备高级法律推理、案例分析和法律解释能力。\n"
    "请根据以下问题，生成逐步的思路链并回答。\n\n"
    "问题：{}\n"
    "回答：\n"
    "<思路>\n"
    "{}\n"
    "</思路>\n"
    "{}"
)
EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    # 这里的字段名称需要根据您手动创建的数据结构来调整
    # 我们手动创建的数据结构将包含 "Question", "Complex_CoT", "Response" 字段
    qs = examples["Question"]
    cots = examples["Complex_CoT"]
    resps = examples["Response"]
    texts = [train_prompt_style.format(q, cot, resp) + EOS_TOKEN for q, cot, resp in zip(qs, cots, resps)]
    return {"text": texts}

# **修改这里，手动创建包含 10 条样本的数据集**
# 创建一个字典列表，每个字典代表一个样本
data = [
    {"Question": "什么是著作权？", "Complex_CoT": "著作权是文学、艺术和科学作品创作者依法享有的专有权利。它包括人身权和财产权。人身权包括发表权、署名权、修改权、保护作品完整权。财产权包括复制权、发行权、出租权、展览权、表演权、放映权、广播权、信息网络传播权、摄制权、改编权、翻译权、汇编权以及其他应由著作权人享有的权利。", "Response": "著作权是创作者对其文学、艺术和科学作品依法享有的权利，包括人身权和财产权，如发表权、复制权等。"},
    {"Question": "如何申请专利？", "Complex_CoT": "申请专利需要向国家知识产权局提交申请文件。申请文件包括专利请求书、说明书及其摘要、权利要求书等。专利类型包括发明、实用新型和外观设计。申请流程一般包括提交申请、受理、初步审查、实质审查（发明专利）、公告、授权。实用新型和外观设计专利不进行实质审查。", "Response": "申请专利需要向国家知识产权局提交申请文件，包括请求书、说明书、权利要求书等，经过审查后可能获得授权。"},
    {"Question": "合同的有效要件有哪些？", "Complex_CoT": "根据《民法典》，有效的合同应当具备以下条件：1. 行为人具有相应的民事行为能力；2. 意思表示真实；3. 不违反法律、行政法规的强制性规定，不违背公序良俗；4. 法律、行政法规规定应当办理批准、登记等手续的，依照其规定。", "Response": "有效的合同需要满足行为人有民事行为能力、意思表示真实、不违反法律强制性规定和公序良俗等条件。"},
    {"Question": "夫妻共同财产包括什么？", "Complex_CoT": "根据《民法典》规定，夫妻在婚姻关系存续期间所得的下列财产，为夫妻的共同财产，归夫妻共同所有：1. 工资、奖金、劳务报酬；2. 生产、经营、投资的收益；3. 知识产权的收益；4. 继承或者受赠的财产，但是本法第一千零六十三条第三项规定的除外；5. 其他应当归共同所有的财产。夫妻对共同财产，有平等的处理权。", "Response": "夫妻共同财产包括婚姻期间所得的工资、奖金、经营收益、知识产权收益、继承或受赠的财产等。"},
    {"Question": "犯罪中止的构成要件是什么？", "Complex_CoT": "犯罪中止是指犯罪分子在实施犯罪过程中，自动放弃犯罪或者自动有效地防止犯罪结果发生。其构成要件包括：1. 发生在犯罪过程中；2. 自动性，即出于犯罪分子本人的意愿而不是由于外部强制力；3. 彻底性，即彻底停止犯罪行为；4. 有效性，即自动采取措施有效地防止了犯罪结果的发生。", "Response": "犯罪中止要求在犯罪过程中，犯罪分子自动、彻底、有效地停止犯罪行为，防止犯罪结果发生。"},
    {"Question": "法人和非法人组织的区别是什么？", "Complex_CoT": "法人是具有民事权利能力和民事行为能力，依法独立享有民事权利和承担民事义务的组织。非法人组织是不具有法人资格，但是能够依法以自己的名义从事民事活动的组织。主要区别在于是否具有独立法人资格，法人拥有独立的法律人格和财产，可以独立承担责任，非法人组织则没有独立的法人人格。", "Response": "法人拥有独立的法律人格和财产，能独立承担责任，非法人组织则不具备独立的法人资格。"},
    {"Question": "民事诉讼的级别管辖如何确定？", "Complex_CoT": "民事诉讼的级别管辖是根据案件的性质、标的额、复杂程度等因素，确定由哪一级人民法院管辖。基层人民法院管辖第一审民事案件，但本法另有规定的除外。中级人民法院、高级人民法院、最高人民法院也都有其各自管辖的第一审民事案件范围，例如涉外案件、有重大影响的案件等。", "Response": "民事诉讼的级别管辖根据案件性质、金额、复杂程度等确定由基层、中级、高级或最高人民法院审理。"},
    {"Question": "什么是行政许可？", "Complex_CoT": "行政许可是指行政机关根据公民、法人或者其他组织的申请，经依法审查，准予其从事特定活动的行为。它是行政机关管理经济和社会事务的一种重要方式，目的在于维护公共利益和社会秩序。行政许可的设定和实施必须依照法定权限、范围、条件和程序进行。", "Response": "行政许可是行政机关依法对特定活动申请进行审查，并决定是否准予的行政行为。"},
    {"Question": "不动产物权的设立、变更、转让和消灭，何时发生效力？", "Complex_CoT": "根据《民法典》规定，不动产物权的设立、变更、转让和消灭，依照法律规定应当登记的，自记载于不动产登记簿时发生效力。未经登记，不发生效力，但是法律另有规定的除外。动产物权的设立和转让，自交付时发生效力。", "Response": "不动产物权的设立、变更等通常自记载于不动产登记簿时发生效力。"},
    {"Question": "什么是知识产权侵权？", "Complex_CoT": "知识产权侵权是指未经知识产权权利人许可，或者没有法律依据，实施了法律禁止的行为，侵害了知识产权权利人依法享有的专有权利的行为。常见的侵权行为包括未经许可复制、发行、信息网络传播他人作品，或者未经许可实施他人专利等。", "Response": "知识产权侵权指未经权利人许可，非法使用其享有知识产权的作品、专利、商标等行为。"}
]

# 将列表数据转换为 datasets 库的 Dataset 对象
dataset = Dataset.from_list(data)

# 格式化数据集
# 注意：因为我们手动创建的数据已经包含了 "Question", "Complex_CoT", "Response" 字段，
# formatting_prompts_func 函数可以直接使用这些字段。
dataset = dataset.map(formatting_prompts_func, batched=True)

print(f"手动创建并格式化后的数据集大小：{len(dataset)}")
print("示例数据（格式化后）：", dataset["text"][0]) # 检查第一个样本格式

# ... 后面的 LoRA 微调和训练代码保持不变 ...

# 3. 使用 LoRA 微调模型
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407
)

# 4. 配置训练参数

batch_size = 2
gradient_accumulation_steps = 4
effective_batch_size = batch_size * gradient_accumulation_steps
# 对于小数据集和测试，设置一个明确的较小 max_steps 是合适的
max_steps = 200 # 例如，只训练 50 步用于快速测试

training_args = TrainingArguments(
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=5,
    max_steps=max_steps,
    learning_rate=5e-5,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    # 如果您想用多 GPU 测试，这些参数是相关的
    # ddp_find_unused_parameters=False,
    # num_gpu=torch.cuda.device_count(), # 如果需要指定，但通常torchrun会自动处理
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=training_args
)

# 5. 开始训练并保存模型

print(f"开始训练，步数：{max_steps}")
# 使用 train 方法时，Trainer会自动管理多GPU（如果torchrun启动）
trainer.train()

# 对于这么小的训练步数，保存模型可能意义不大，但代码结构保留
new_model_local = "DeepSeek-R1-Legal-COT-merged_test" # 修改保存目录名称以区分
model.save_pretrained_merged(new_model_local, tokenizer, save_method="merged_16bit") # 保存模型
tokenizer.save_pretrained(new_model_local)  # 保存分词器
print(f"模型已保存至：{new_model_local}")  # 输出保存路径
