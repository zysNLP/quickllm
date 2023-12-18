#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
@Time    : 2023/10/18 21:01
@Author  : nijiahui
@FileName: llm.py
@Software: PyCharm
 
'''

from fastapi import FastAPI,HTTPException
from fastapi.responses import StreamingResponse,JSONResponse
from pydantic import BaseModel
from typing import List, Dict
import requests
from logger_config import logger  # 引入你的 logger
import openai
from typing import Literal

from request_id_generator import generate_request_id
from config import LLM_13B_ENGINE_DICT, LLM_70B_ENGINE_DICT, EMBEDDING_ENGINE_DICT
import config

from transformers import BertTokenizer
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
import numpy as np
from concurrent.futures import ThreadPoolExecutor

LLM_ENGINE_DICT = LLM_13B_ENGINE_DICT

app = FastAPI()

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class TokenRequest(BaseModel):
    messages: list[Message]    # 对话消息
    stream: bool = True        # 默认值为 True

class EmbeddingRequests(BaseModel):
    strings: List[str]
        
class EmbeddingRequest(BaseModel):
    strings: str
        
# 在应用启动时执行的初始化逻辑
@app.on_event("startup")
def startup_event():
    # 1、init embedding_tokenizer
    global embedding_tokenizer
    embedding_tokenizer = BertTokenizer.from_pretrained(EMBEDDING_ENGINE_DICT["model_path"])
    
    # 2、init embedding_model_client
    global embedding_model_client
    embedding_model_client = grpcclient.InferenceServerClient(url=EMBEDDING_ENGINE_DICT["triton_url"])

    # 3、使用 ThreadPoolExecutor 来异步执行同步推理函数
    global embedding_executor
    embedding_executor = ThreadPoolExecutor(max_workers=128)
    
# embedding 服务
@app.post("/v1/text/embeddings")
async def text_embedding(request: EmbeddingRequests):
    # 假设这是您的同步推理函数
    def sync_infer(model_name, inputs, outputs):
        return embedding_model_client.infer(model_name, inputs, outputs=outputs)
    
    def data_tokenizer(text):
        # 编码文本
        encoded_inputs = embedding_tokenizer(
            text,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
            max_length=EMBEDDING_ENGINE_DICT["max_length"]
        )
        # 创建输入数据的字典
        inputs = [
            grpcclient.InferInput('input_ids', [len(text), EMBEDDING_ENGINE_DICT["max_length"]], "INT32"),
            grpcclient.InferInput('attention_mask', [len(text), EMBEDDING_ENGINE_DICT["max_length"]], "INT32"),
            grpcclient.InferInput('token_type_ids', [len(text), EMBEDDING_ENGINE_DICT["max_length"]], "INT32"),
        ]
        # 设置数据
        inputs[0].set_data_from_numpy(encoded_inputs['input_ids'].numpy().astype('int32'))
        inputs[1].set_data_from_numpy(encoded_inputs['attention_mask'].numpy().astype('int32'))
        inputs[2].set_data_from_numpy(encoded_inputs['token_type_ids'].numpy().astype('int32'))
        # 定义输出
        outputs = [
            # grpcclient.InferRequestedOutput('output_0'),
            grpcclient.InferRequestedOutput('output_1'),
        ]
        return inputs, outputs

    try: 
        embedding_res_list = list()
        tasks = []
        
        for text in request.strings:
            inputs, outputs = data_tokenizer([text])
            # 将同步推理函数包装为异步执行的任务
            task = embedding_executor.submit(sync_infer, EMBEDDING_ENGINE_DICT["model_name"], inputs, outputs)
            tasks.append(task)

        # 等待所有任务完成
        for future in tasks:
            embedding_results = future.result()
            embedding_res_list.append(np.squeeze(embedding_results.as_numpy('output_1')).tolist())

        content = {"embeddings": embedding_res_list}
        return JSONResponse(content=content, status_code=200)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# embedding 服务
@app.post("/v1/text/embedding")
def text_embedding(request: EmbeddingRequest):
    def data_tokenizer(text):
        # 编码文本
        encoded_inputs = embedding_tokenizer(
            text,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
            max_length=EMBEDDING_ENGINE_DICT["max_length"]
        )
        # 创建输入数据的字典
        inputs = [
            grpcclient.InferInput('input_ids', [1, EMBEDDING_ENGINE_DICT["max_length"]], "INT32"),
            grpcclient.InferInput('attention_mask', [1, EMBEDDING_ENGINE_DICT["max_length"]], "INT32"),
            grpcclient.InferInput('token_type_ids', [1, EMBEDDING_ENGINE_DICT["max_length"]], "INT32"),
        ]
        # 设置数据
        inputs[0].set_data_from_numpy(encoded_inputs['input_ids'].numpy().astype('int32'))
        inputs[1].set_data_from_numpy(encoded_inputs['attention_mask'].numpy().astype('int32'))
        inputs[2].set_data_from_numpy(encoded_inputs['token_type_ids'].numpy().astype('int32'))
        # 定义输出
        outputs = [
            # grpcclient.InferRequestedOutput('output_0'),
            grpcclient.InferRequestedOutput('output_1'),
        ]
        return inputs, outputs

    try: 
        inputs, outputs = data_tokenizer(request.dict()["strings"])
        # 发送推理请求
        embedding_results = embedding_model_client.infer(EMBEDDING_ENGINE_DICT["model_name"], inputs,outputs=outputs)
        content = {"embeddings": embedding_results.as_numpy('output_1').tolist()}
        return JSONResponse(content=content, status_code=200)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
# 大模型服务
@app.post("/v1/chat/completion")
def openai_chat_completion(request: TokenRequest):
    """ Chat Completion API """
    try:
        # 生成 request 唯一 ID
        request_id = generate_request_id()

        # 获取基础信息
        logger.info(f"request_id:{request_id}; 【start】进入逻辑")
        logger.info(f"request_id:{request_id}; 【request】{request.dict()}")

        # prompt 
        prompts = '\n'.join([f'{i["role"]}: {i["content"]}' for i in request.dict()["messages"]]) + "\n assistant: "
        prompts = LLM_ENGINE_DICT["SYSTEM"] + prompts
        logger.info(f"request_id:{request_id}; 【In-info】prompt: {prompts}")

        # openai 配置项
        openai.api_key = LLM_ENGINE_DICT["api_key"]
        openai.api_base = LLM_ENGINE_DICT["api_base"]

        def get_response(request):
            response = openai.Completion.create(
                **LLM_ENGINE_DICT["requests_parameter"],
                prompt=prompts,
                stream=request.stream
            )
            if request.stream:
                logger.info(f"request_id:{request_id}; stream generate start")
                for data in response:
                    chunk = data["choices"][0]["text"]
                    print(chunk, end="", flush=True)
                    yield chunk
                logger.info(f"request_id:{request_id}; stream generate finished")
            else:
                if response.choices:
                    answer = response.choices[0]["text"]
                    print(f"request_id:{request_id}; no-stream generate finished:{answer}")
                    logger.info(f"request_id:{request_id}; no-stream generate finished:{answer}")
                    yield(answer)
        return StreamingResponse(
            get_response(request),
            media_type='text/event-stream',
        )
    except Exception as e:
        logger.error(f"request_id:{request_id}; 【finally】: failed；{e}")
        print(f"request_id:{request_id}; 【finally】: failed；{e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/chatcompletion")
async def openai_chat_chatcompletion(request: TokenRequest):
    """ Chat Chatcompletion API """
        
    try:
        # 生成 request 唯一 ID
        request_id = generate_request_id()

        # 获取基础信息
        logger.info(f"request_id:{request_id}; 【start】进入逻辑")
        logger.info(f"request_id:{request_id}; 【request】{request.dict()}")

        # openai 配置项
        openai.api_key = LLM_ENGINE_DICT["api_key"]
        openai.api_base = LLM_ENGINE_DICT["api_base"]

        # promp <-> messages
        messages = request.dict()["messages"]
        messages.insert(0,{"role": "system", "content": LLM_ENGINE_DICT["SYSTEM"]})
        logger.info(f"request_id:{request_id}; 【In-info】prompt: {messages}")

        async def get_response(request):
            response = await openai.ChatCompletion.acreate(
                **LLM_ENGINE_DICT["requests_parameter"],
                messages=messages,
                stream=request.stream
            )
            if request.stream:
                logger.info(f"request_id:{request_id}; stream generate start")
                async for data in response:
                    if choices := data.choices:
                        if chunk := choices[0].get("delta", {}).get("content"):
                            print(chunk, end="", flush=True)
                            yield chunk
                logger.info(f"request_id:{request_id}; stream generate finished")
            else:
                if response.choices:
                    answer = response.choices[0].message.content
                    print(f"request_id:{request_id}; no-stream generate finished:{answer}")
                    logger.info(f"request_id:{request_id}; no-stream generate finished:{answer}")
                    yield(answer)
                    
        return StreamingResponse(
            get_response(request),
            media_type='text/event-stream',
        )
    except Exception as e:
        logger.error(f"request_id:{request_id}; 【finally】: failed；{e}")
        print(f"request_id:{request_id}; 【finally】: failed；{e}")
        raise HTTPException(status_code=500, detail=str(e))