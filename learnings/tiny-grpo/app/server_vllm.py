# -*- coding: utf-8 -*-
"""
使用vLLM的推理服务器
"""
import os
import time
import traceback
from fastapi import HTTPException
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

from config import (
    TRAINED_MODEL_PATH, FALLBACK_TOKENIZER_PATH, SYSTEM_PROMPT, 
    VLLM_SAMPLING_PARAMS, VLLM_MODEL_CONFIG, VLLM_PORT
)
from models import QuestionInput, ResponseAttributes
from utils import setup_logging, compute_elapsed_time, parse_reasoning, create_fastapi_app, API_DOCUMENTATION

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 初始化
logger = setup_logging()

# 加载 processor (tokenizer)
try:
    processor = AutoProcessor.from_pretrained(TRAINED_MODEL_PATH)
    logger.info(f"Successfully loaded processor from {TRAINED_MODEL_PATH}")
except Exception as e:
    logger.warning(f"Failed to load processor from {TRAINED_MODEL_PATH}: {e}")
    logger.info("Falling back to original processor path")
    processor = AutoProcessor.from_pretrained(FALLBACK_TOKENIZER_PATH)

# 初始化vLLM
vllm_model = LLM(
    model=TRAINED_MODEL_PATH,
    **VLLM_MODEL_CONFIG
)

# 采样参数
sampling_params = SamplingParams(**VLLM_SAMPLING_PARAMS)

# 创建FastAPI应用
app, mcp = create_fastapi_app()


def run_inference(question: str):
    """推理函数"""
    chat_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    chat_prompt = processor.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    
    outputs = vllm_model.generate([chat_prompt], sampling_params)
    output_text = outputs[0].outputs[0].text
    
    return output_text


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
    return {"message": "Math Reasoning Service (vLLM) is running", "status": "healthy"}


# 挂载 MCP 应用
app.mount("/", mcp.streamable_http_app())


if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=VLLM_PORT, workers=1)
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f"Error starting service: {e}") 