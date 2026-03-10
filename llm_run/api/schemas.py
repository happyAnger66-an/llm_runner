"""
OpenAI 兼容的请求/响应模型定义
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# --- Chat Completions ---
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="模型名称")
    messages: List[ChatMessage] = Field(..., description="对话消息列表")
    stream: bool = Field(default=False, description="是否流式返回")
    max_tokens: Optional[int] = Field(default=512, description="最大生成 token 数")
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2)
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1)
    top_k: Optional[int] = Field(default=50, ge=0)
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(default=0.0)
    frequency_penalty: Optional[float] = Field(default=0.0)
    n: Optional[int] = Field(default=1, description="返回的候选数")
    user: Optional[str] = None


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Optional[Dict[str, int]] = None


# --- Completions (legacy) ---
class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str], List[int]]
    stream: bool = False
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    stop: Optional[Union[str, List[str]]] = None
    n: Optional[int] = 1


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[str] = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Optional[Dict[str, int]] = None


# --- Models List ---
class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "llm_run"


class ModelsListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]
