"""
FastAPI 应用工厂
"""

from typing import Optional

from llm_run.api.routes import router
from llm_run.api.deps import init_engine


def create_app(
    engine_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    api_prefix: str = "/v1",
):
    """创建 FastAPI 应用"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="LLM Run API",
        description="OpenAI 兼容的 LLM 推理 REST API",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix=api_prefix)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    if engine_path:
        init_engine(engine_path, tokenizer_path=tokenizer_path)

    return app
