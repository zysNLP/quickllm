# -*- coding: utf-8 -*-
"""
公共工具函数
"""
import time
import re
import logging
from typing import Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.fastmcp import FastMCP
from config import SERVICE_NAME, LOG_FORMAT, LOG_LEVEL


def setup_logging():
    """设置日志"""
    logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
    return logging.getLogger(__name__)


def compute_elapsed_time(start_time: float) -> float:
    """计算elapsed时间（毫秒）"""
    end_time = time.time()
    return round((end_time - start_time) * 1000, 2)


def parse_reasoning(raw_output: str) -> Dict[str, str]:
    """
    解析推理结果，提取不同部分
    """
    # 现在raw_output就是assistant的回复
    assistant_response = raw_output.strip()
    
    # 提取 <think> </think> 内的推理过程
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, assistant_response, re.DOTALL)
    reasoning_process = think_match.group(1).strip() if think_match else ""
    
    # 提取 <answer> </answer> 内的最终答案
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, assistant_response, re.DOTALL)
    final_answer = answer_match.group(1).strip() if answer_match else ""
    
    # 如果没有找到标准格式的标签，尝试其他方式提取
    if not final_answer:
        # 查找可能的答案模式
        lines = assistant_response.split('\n')
        for line in lines:
            if '<answer>' in line and '</answer>' in line:
                final_answer = line.split('<answer>')[1].split('</answer>')[0].strip()
                break
    
    # 移除标签后的助手回复（用于显示）
    clean_response = assistant_response
    if think_match:
        clean_response = clean_response.replace(think_match.group(0), "").strip()
    if answer_match:
        clean_response = clean_response.replace(answer_match.group(0), "").strip()
    
    return {
        "assistant_response": assistant_response,
        "reasoning_process": reasoning_process,
        "final_answer": final_answer,
        "clean_response": clean_response
    }


def create_fastapi_app():
    """创建FastAPI应用和MCP"""
    # MCP 初始化
    mcp = FastMCP(name=SERVICE_NAME, stateless_http=True)

    # FastAPI 生命周期管理
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with mcp.session_manager.run():
            yield

    app = FastAPI(openapi_url=f"/{SERVICE_NAME}/openapi.json", lifespan=lifespan)

    # CORS 设置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app, mcp


# API文档模板
API_DOCUMENTATION = """
    ## 1. Name
    **Math Reasoning Service**

    ## 2. Description
    This API solves mathematical reasoning questions using a trained language model. The assistant provides a step-by-step reasoning process and a final answer.

    ## 3. Input Parameters
    ### 3.1 Request Body Structure
    - **`QuestionInput`** *(required)*: Contains the question to be solved.
      - **`question`** *(str, required)*: The mathematical reasoning question.

    ## 4. Return Value
    - **`ResponseAttributes`**: Contains the reasoning result and metadata.
      - **`code`** *(int)*: HTTP status code.
        - **`200`**: Success.
        - **`400`**: Invalid request.
        - **`500`**: Processing failed.
      - **`data`** *(dict)*: Analysis results.
        - **`assistant_response`** *(str)*: Full assistant response.
        - **`reasoning_process`** *(str)*: Step-by-step reasoning extracted from <think> tags.
        - **`final_answer`** *(str)*: Final answer extracted from <answer> tags.
        - **`clean_response`** *(str)*: Assistant response with tags removed.
      - **`elapsed_milliseconds`** *(float)*: Processing time in milliseconds.
      - **`message`** *(str)*: Success/error message.

    ## 5. Examples
    ### Request Example
    ```json
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    }
    ```
    ### Response Example
    ```json
    {
        "code": 200,
        "elapsed_milliseconds": 123.45,
        "data": {
            "assistant_response": "In April, Natalia sold clips to 48 friends. In May, she sold half as many clips as in April, which is 1/2 × 48 = 24 clips. Therefore, the total number of clips sold in April and May is 48 + 24 = 72. So, Natalia sold 72 clips altogether in April and May. <answer>72</answer>",
            "reasoning_process": "In April, Natalia sold clips to 48 friends. In May, she sold half as many clips as in April, which is 1/2 × 48 = 24 clips. Therefore, the total number of clips sold in April and May is 48 + 24 = 72. So, Natalia sold 72 clips altogether in April and May.",
            "final_answer": "72",
            "clean_response": "In April, Natalia sold clips to 48 friends. In May, she sold half as many clips as in April, which is 1/2 × 48 = 24 clips. Therefore, the total number of clips sold in April and May is 48 + 24 = 72. So, Natalia sold 72 clips altogether in April and May."
        },
        "message": "Question solved successfully"
    }
    ```
    """ 