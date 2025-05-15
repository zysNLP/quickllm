# -*- coding: utf-8 -*-
"""
    @Project : quickllm
    @File    : server_vllm.py
    @Author  : sunday
    @Time    : 2025/5/15 10:00
"""

from fastapi import FastAPI
from pydantic import BaseModel
import httpx

app = FastAPI()
VLLM_API_URL = "http://localhost:8100/v1/chat/completions"  # vllm官方API Server地址

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(request: PromptRequest):
    # 构造OpenAI格式请求
    payload = {
        "model": "R1-14B",  # 可随便填，vllm会忽略
        "messages": [
            {"role": "user", "content": request.prompt}
        ],
        "max_tokens": 128,
        "temperature": 0.8,
        "top_p": 0.95
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(VLLM_API_URL, json=payload, timeout=60)
        data = resp.json()
        # 解析OpenAI格式返回
        result = data["choices"][0]["message"]["content"]
        return {"result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8101)