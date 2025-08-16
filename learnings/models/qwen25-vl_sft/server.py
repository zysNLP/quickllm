# -*- coding: utf-8 -*-
"""
    @Project : quickllm
    @File    : server.py
    @Author  : sunday
    @Time    : 2025/5/16 13:25
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import traceback
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import asyncio

app = FastAPI()

# 初始化模型和参数
MODEL_PATH = "/quickllm/LLaMA-Factory/qwen2.5-mmlm0513"
llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 10, "video": 10},
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=256,
    stop_token_ids=[],
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)


class Message(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):
    batch_messages: List[List[Message]]  # 这里是batch，每个元素是一组对话


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # 处理batch
        llm_inputs_list = []
        for messages in request.batch_messages:
            messages_dict = [m.model_dump() for m in messages]
            prompt = processor.apply_chat_template(
                messages_dict,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs, _ = process_vision_info(messages_dict, return_video_kwargs=True)
            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs
            llm_inputs = {
                "prompt": prompt,
                "multi_modal_data": mm_data,
            }
            llm_inputs_list.append(llm_inputs)

        outputs = llm.generate(llm_inputs_list, sampling_params=sampling_params)

        results = [out.outputs[0].text for out in outputs]
        return {"responses": results}
    except Exception as e:
        msg = traceback.format_exc()
        print(f"Error: {msg}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7868)
