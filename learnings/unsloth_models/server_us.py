import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from unsloth import FastLanguageModel

# 读取环境变量，决定当前进程用哪块GPU
cuda_id = os.environ.get("CUDA_VISIBLE_DEVICES", "3")
print(f"当前进程使用的GPU: {cuda_id}")

merged_model_dir = "DeepSeek-R1-Legal-COT-merged_test"
max_seq_length = 2048

if not os.path.exists(merged_model_dir):
    raise FileNotFoundError(f"模型目录 {merged_model_dir} 不存在")

model, tokenizer = FastLanguageModel.from_pretrained(
    merged_model_dir,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True
)
FastLanguageModel.for_inference(model)
device = "cuda" if torch.cuda.is_available() else "cpu"

prompt_template = (
    "你是一位法律专家，具备高级法律推理、案例分析和法律解释能力。\n"
    "请根据以下问题，生成逐步的思路链并回答。\n\n"
    "问题：{}\n"
    "回答：\n"
    "<思路>{}</思路>\n"
)

app = FastAPI()

class BatchRequestItem(BaseModel):
    questions: List[str]

class BatchResponseItem(BaseModel):
    code: int
    data: dict
    message: str
    elapsed: float

@app.post("/legal_cot_batch", response_model=BatchResponseItem)
async def legal_cot_batch(request: BatchRequestItem):
    import time
    start = time.time()
    questions = [q.strip() for q in request.questions if q.strip()]
    if not questions:
        return BatchResponseItem(code=400, data={}, message="问题不能为空", elapsed=0)
    try:
        input_texts = [prompt_template.format(q, "") for q in questions]
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length).to(device)
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1000,
            temperature=0.7,
            top_p=0.9,
            use_cache=True
        )
        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # 去掉prompt部分
        clean_responses = [
            resp[len(input_texts[i]):].strip() if resp.startswith(input_texts[i]) else resp
            for i, resp in enumerate(responses)
        ]
        elapsed = round((time.time() - start) * 1000, 2)
        return BatchResponseItem(
            code=200,
            data={"responses": clean_responses},
            message="success",
            elapsed=elapsed
        )
    except Exception as e:
        elapsed = round((time.time() - start) * 1000, 2)
        return BatchResponseItem(code=500, data={}, message=str(e), elapsed=elapsed)

if __name__ == "__main__":
    import uvicorn
    # 你可以用如下命令行启动多进程多卡（见下方说明）
    uvicorn.run("server_us:app", host="0.0.0.0", port=8100, workers=1)