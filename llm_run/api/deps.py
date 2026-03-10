"""
API 依赖注入
"""

from typing import Optional

from llm_run.engine.base import BaseInferenceEngine
from llm_run.engine.tensorrt_engine import TensorRTEngine


_engine: Optional[BaseInferenceEngine] = None


def get_engine() -> BaseInferenceEngine:
    """获取全局推理引擎实例"""
    if _engine is None:
        raise RuntimeError("推理引擎未初始化，请在启动时调用 init_engine()")
    return _engine


def init_engine(engine_path: str, tokenizer_path: Optional[str] = None, **kwargs) -> BaseInferenceEngine:
    """初始化全局推理引擎"""
    global _engine
    _engine = TensorRTEngine()
    _engine.load(engine_path, tokenizer_path=tokenizer_path, **kwargs)
    return _engine


def set_engine(engine: BaseInferenceEngine) -> None:
    """设置自定义引擎实例（用于测试或替换实现）"""
    global _engine
    _engine = engine
