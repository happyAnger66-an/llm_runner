"""
推理引擎抽象基类
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional


class BaseInferenceEngine(ABC):
    """推理引擎抽象基类，定义统一的推理接口"""

    @abstractmethod
    def load(self, engine_path: str, **kwargs) -> None:
        """
        加载 TensorRT engine

        Args:
            engine_path: engine 文件或目录路径
            **kwargs: 其他加载参数（如 tokenizer_path 等）
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 50,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:
        """
        批量生成文本

        Args:
            prompts: 输入提示列表
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_p: nucleus sampling 参数
            top_k: top-k sampling 参数
            stop: 停止词列表
            **kwargs: 其他生成参数

        Returns:
            生成的文本列表
        """
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 50,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        流式生成文本

        Args:
            prompt: 输入提示
            其他参数同 generate

        Yields:
            生成的 token 片段
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """当前加载的模型名称"""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """engine 是否已加载"""
        pass
