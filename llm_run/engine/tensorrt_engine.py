"""
TensorRT-LLM 推理引擎实现
"""

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, List, Optional

from llm_run.engine.base import BaseInferenceEngine


class TensorRTEngine(BaseInferenceEngine):
    """
    基于 TensorRT-LLM 的推理引擎

    支持加载预编译的 TensorRT engine 并执行推理。
    engine_path 可以是：
    - 单个 .engine 文件路径
    - 包含 engine 文件的目录（TensorRT-LLM 多 rank 格式）
    """

    def __init__(self):
        self._llm = None
        self._tokenizer = None
        self._engine_path: Optional[str] = None
        self._model_name: str = "unknown"

    def load(self, engine_path: str, tokenizer_path: Optional[str] = None, **kwargs) -> None:
        """
        加载 TensorRT engine

        Args:
            engine_path: engine 文件或目录路径
            tokenizer_path: tokenizer 路径，若为 None 则尝试从 engine 同目录加载
            **kwargs: 传递给 tensorrt_llm.LLM 的其他参数
        """
        try:
            from tensorrt_llm import LLM
        except ImportError:
            raise ImportError(
                "请安装 tensorrt_llm: pip install tensorrt_llm\n"
                "或从 NVIDIA 官方渠道获取 TensorRT-LLM wheel"
            )

        path = Path(engine_path)
        if not path.exists():
            raise FileNotFoundError(f"Engine 路径不存在: {engine_path}")

        # TensorRT-LLM 支持从目录加载预编译 engine
        self._llm = LLM(
            model_dir=str(path) if path.is_dir() else str(path.parent),
            tokenizer_dir=tokenizer_path or str(path.parent),
            **kwargs,
        )
        self._engine_path = engine_path
        self._model_name = path.stem if path.is_file() else path.name

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
        """批量生成"""
        if self._llm is None:
            raise RuntimeError("请先调用 load() 加载 engine")

        from tensorrt_llm import SamplingParams

        sampling_params = SamplingParams(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop or [],
            **{k: v for k, v in kwargs.items() if k in ("repetition_penalty", "length_penalty")},
        )

        outputs = self._llm.generate(prompts, sampling_params)
        return [out.outputs[0].text for out in outputs]

    async def generate_stream(
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
        """流式生成（在线程池中运行同步生成并逐 token 返回）"""
        if self._llm is None:
            raise RuntimeError("请先调用 load() 加载 engine")

        from tensorrt_llm import SamplingParams

        sampling_params = SamplingParams(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop or [],
        )

        loop = asyncio.get_event_loop()

        def _run():
            for output in self._llm.generate([prompt], sampling_params, streaming=True):
                for text in output.outputs[0].token_ids:
                    yield text

        # 简化实现：先同步生成完整结果再流式返回
        # 实际生产可接入 tensorrt_llm 的 streaming 接口
        def _sync_gen():
            out = self._llm.generate([prompt], sampling_params)
            return out[0].outputs[0].text

        text = await loop.run_in_executor(None, _sync_gen)
        for char in text:
            yield char

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None
