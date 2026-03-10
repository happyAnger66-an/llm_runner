"""
OpenAI 兼容 API 路由
"""

import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from llm_run.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    ModelsListResponse,
    ModelInfo,
)
from llm_run.api.deps import get_engine
from llm_run.engine.base import BaseInferenceEngine

router = APIRouter()


def _format_messages_to_prompt(messages: list) -> str:
    """将 OpenAI 格式的 messages 转为模型输入 prompt"""
    parts = []
    for msg in messages:
        role = msg.role
        content = msg.content or ""
        if role == "system":
            parts.append(f"<|system|>\n{content}\n")
        elif role == "user":
            parts.append(f"<|user|>\n{content}\n")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


def _parse_stop(stop):
    """将 stop 参数转为 list"""
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    return list(stop)


# --- /v1/models ---
@router.get("/models", response_model=ModelsListResponse)
async def list_models(engine: BaseInferenceEngine = Depends(get_engine)):
    """列出可用模型（OpenAI 兼容）"""
    if not engine.is_loaded:
        return ModelsListResponse(data=[])
    return ModelsListResponse(
        data=[ModelInfo(id=engine.model_name, owned_by="llm_run")]
    )


# --- /v1/chat/completions ---
@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    engine: BaseInferenceEngine = Depends(get_engine),
):
    """Chat Completions 接口（OpenAI 兼容）"""
    if not engine.is_loaded:
        raise HTTPException(503, "模型未加载，请先配置并加载 engine")

    prompt = _format_messages_to_prompt(request.messages)
    stop = _parse_stop(request.stop)

    if request.stream:
        return StreamingResponse(
            _stream_chat_completion(engine, request.model, prompt, request, stop),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # 非流式
    outputs = engine.generate(
        [prompt],
        max_new_tokens=request.max_tokens or 512,
        temperature=request.temperature or 0.7,
        top_p=request.top_p or 1.0,
        top_k=request.top_k or 50,
        stop=stop,
    )
    text = outputs[0] if outputs else ""

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )


async def _stream_chat_completion(engine, model: str, prompt: str, request, stop):
    """流式返回 SSE 格式"""
    import json

    async for chunk in engine.generate_stream(
        prompt,
        max_new_tokens=request.max_tokens or 512,
        temperature=request.temperature or 0.7,
        top_p=request.top_p or 1.0,
        top_k=request.top_k or 50,
        stop=stop,
    ):
        data = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"


# --- /v1/completions ---
@router.post("/completions", response_model=CompletionResponse)
async def completions(
    request: CompletionRequest,
    engine: BaseInferenceEngine = Depends(get_engine),
):
    """Completions 接口（OpenAI 兼容，legacy）"""
    if not engine.is_loaded:
        raise HTTPException(503, "模型未加载")

    prompt = request.prompt
    if isinstance(prompt, list):
        if prompt and isinstance(prompt[0], int):
            raise HTTPException(400, "token ids 输入暂不支持")
        prompt = prompt[0] if isinstance(prompt[0], str) else str(prompt)
    elif not isinstance(prompt, str):
        prompt = str(prompt)

    stop = _parse_stop(request.stop)
    outputs = engine.generate(
        [prompt],
        max_new_tokens=request.max_tokens or 512,
        temperature=request.temperature or 0.7,
        top_p=request.top_p or 1.0,
        stop=stop,
    )
    text = outputs[0] if outputs else ""

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=request.model,
        choices=[CompletionChoice(index=0, text=text, finish_reason="stop")],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )
