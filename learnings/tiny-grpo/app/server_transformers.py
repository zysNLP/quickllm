# -*- coding: utf-8 -*-
"""
使用Transformers的推理服务器
"""
import os
import time
import traceback
import torch
from fastapi import HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    TRAINED_MODEL_PATH, FALLBACK_TOKENIZER_PATH, SYSTEM_PROMPT, 
    GENERATION_CONFIG, RANDOM_SEED, TRANSFORMERS_PORT
)
from models import QuestionInput, ResponseAttributes
from utils import setup_logging, compute_elapsed_time, parse_reasoning, create_fastapi_app, API_DOCUMENTATION

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 初始化
logger = setup_logging()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 tokenizer 和训练后的模型
try:
    tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_PATH)
    logger.info(f"Successfully loaded tokenizer from {TRAINED_MODEL_PATH}")
except Exception as e:
    logger.warning(f"Failed to load tokenizer from {TRAINED_MODEL_PATH}: {e}")
    logger.info("Falling back to original tokenizer path")
    tokenizer = AutoTokenizer.from_pretrained(FALLBACK_TOKENIZER_PATH)

tokenizer.pad_token = tokenizer.eos_token

def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()
    return model

trained_model = load_model(TRAINED_MODEL_PATH)

# 更新生成配置，添加tokenizer的pad_token_id
generation_config = {
    **GENERATION_CONFIG,
    "pad_token_id": tokenizer.eos_token_id,
}

# 创建FastAPI应用
app, mcp = create_fastapi_app()


def run_inference(question: str):
    """推理函数"""
    # 设置随机种子以确保可重复性
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
    
    chat_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = trained_model.generate(**inputs, **generation_config)
        # 只解码新生成的部分，排除输入的tokens
        generated_ids = output_ids[0][len(inputs.input_ids[0]):]
        output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return output


@mcp.tool()
@app.post("/solve_question", response_model=ResponseAttributes, operation_id="solve_question")
async def solve_question(input_data: QuestionInput):
    API_DOCUMENTATION
    start_time = time.time()
    try:
        if not input_data.question:
            raise ValueError("No question provided.")

        reasoning = run_inference(input_data.question)
        logger.info(f"reasoning: \n{reasoning}")
        parsed_result = parse_reasoning(reasoning)
        code = 200
        message = "Question solved successfully"

        return ResponseAttributes(
            code=code,
            elapsed_milliseconds=compute_elapsed_time(start_time),
            data=parsed_result,
            message=message
        )
    except ValueError as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in question solving: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Unexpected error occurred.")


@app.get("/")
async def index():
    return {"message": "Math Reasoning Service (Transformers) is running", "status": "healthy"}


# 挂载 MCP 应用
app.mount("/", mcp.streamable_http_app())


if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=TRANSFORMERS_PORT, workers=1)
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f"Error starting service: {e}") 