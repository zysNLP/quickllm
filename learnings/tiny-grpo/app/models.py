# -*- coding: utf-8 -*-
"""
数据模型定义
"""
from pydantic import BaseModel
from typing import Dict


class QuestionInput(BaseModel):
    """问题输入模型"""
    question: str


class ResponseAttributes(BaseModel):
    """响应属性模型"""
    code: int
    elapsed_milliseconds: float
    data: Dict[str, str]
    message: str 