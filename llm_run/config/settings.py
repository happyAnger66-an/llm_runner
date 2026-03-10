"""
应用配置
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置"""

    # 模型配置
    engine_path: Optional[str] = Field(default=None, description="TensorRT engine 路径")
    tokenizer_path: Optional[str] = Field(default=None, description="Tokenizer 路径")
    model_name: str = Field(default="llm", description="模型显示名称")

    # API 配置
    host: str = Field(default="0.0.0.0", description="API 监听地址")
    port: int = Field(default=8000, description="API 端口")
    api_prefix: str = Field(default="/v1", description="API 路径前缀")

    # 推理默认参数
    max_new_tokens: int = Field(default=512, description="默认最大生成 token 数")
    temperature: float = Field(default=0.7, description="默认温度")
    top_p: float = Field(default=1.0, description="默认 top_p")
    top_k: int = Field(default=50, description="默认 top_k")

    class Config:
        env_prefix = "LLM_RUN_"
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()
