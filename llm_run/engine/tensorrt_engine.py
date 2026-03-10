"""
基于原始 TensorRT 的推理引擎实现

适用于 Jetson Thor 等嵌入式环境，仅依赖 tensorrt 和 pycuda，
不依赖 tensorrt_llm。
"""

import asyncio
from pathlib import Path
from typing import AsyncIterator, List, Optional

import numpy as np

from llm_run.engine.base import BaseInferenceEngine


def _sample_next_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
) -> int:
    """从 logits 中采样下一个 token"""
    if temperature <= 0:
        return int(np.argmax(logits))

    logits = logits.astype(np.float32) / temperature
    logits = logits - np.max(logits)
    probs = np.exp(logits) / np.sum(np.exp(logits))

    if top_k > 0:
        indices = np.argpartition(probs, -top_k)[-top_k:]
        probs = probs[indices]
        probs = probs / probs.sum()
        idx = np.random.choice(len(probs), p=probs)
        return int(indices[idx])

    if top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[sorted_indices])
        mask = cumsum <= top_p
        mask[0] = True
        probs = probs[sorted_indices]
        probs[~mask] = 0
        probs = probs / probs.sum()
        idx = np.random.choice(len(probs), p=probs)
        return int(sorted_indices[idx])

    return int(np.random.choice(len(probs), p=probs))


class TensorRTEngine(BaseInferenceEngine):
    """
    基于原始 TensorRT 的推理引擎

    支持 Jetson Thor 等嵌入式环境，依赖：
    - tensorrt（JetPack 自带或 pip install tensorrt）
    - pycuda（pip install pycuda）
    - transformers（用于 tokenizer，pip install transformers）

    兼容 TensorRT 8.x 和 10.x API。
    """

    def __init__(self):
        self._engine = None
        self._context = None
        self._tokenizer = None
        self._stream = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None
        self._engine_path: Optional[str] = None
        self._model_name: str = "unknown"
        self._vocab_size: int = 0
        self._pad_id: int = 0
        self._eos_id: int = 0
        self._use_v2_api: bool = True  # True: 8.x execute_async_v2, False: 10.x execute_async_v3

    def load(
        self,
        engine_path: str,
        tokenizer_path: Optional[str] = None,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        加载 TensorRT engine

        Args:
            engine_path: .engine 或 .plan 文件路径
            tokenizer_path: tokenizer 路径或 HuggingFace 模型名
            input_name: 输入 binding 名称，None 则自动检测
            output_name: 输出 binding 名称，None 则自动检测
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "请安装 tensorrt 和 pycuda:\n"
                "  pip install tensorrt pycuda\n"
                "Jetson 上 tensorrt 通常随 JetPack 预装"
            ) from e

        path = Path(engine_path)
        if not path.exists():
            raise FileNotFoundError(f"Engine 路径不存在: {engine_path}")

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        if self._engine is None:
            raise RuntimeError("Engine 反序列化失败")

        self._context = self._engine.create_execution_context()

        # 兼容 TensorRT 8.x 与 10.x
        if hasattr(self._engine, "num_io_tensors"):
            # TensorRT 10.x
            self._use_v2_api = False
            num_bindings = self._engine.num_io_tensors
            names = [self._engine.get_tensor_name(i) for i in range(num_bindings)]
            input_names = [
                n for n in names
                if self._engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
            ]
            output_names = [
                n for n in names
                if self._engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT
            ]
        else:
            # TensorRT 8.x
            self._use_v2_api = True
            names = [self._engine.get_binding_name(i) for i in range(self._engine.num_bindings)]
            input_names = [n for i, n in enumerate(names) if self._engine.binding_is_input(i)]
            output_names = [n for i, n in enumerate(names) if not self._engine.binding_is_input(i)]

        self._input_name = input_name or (input_names[0] if input_names else None)
        self._output_name = output_name or (output_names[0] if output_names else None)

        if not self._input_name or not self._output_name:
            raise RuntimeError(f"无法确定 input/output binding，可用: {names}")

        self._stream = cuda.Stream()
        self._load_tokenizer(tokenizer_path or str(path.parent))
        self._engine_path = engine_path
        self._model_name = path.stem

    def _load_tokenizer(self, path: str) -> None:
        """加载 tokenizer"""
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("请安装 transformers: pip install transformers")

        self._tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self._vocab_size = len(self._tokenizer)
        self._pad_id = self._tokenizer.pad_token_id or 0
        self._eos_id = (
            self._tokenizer.eos_token_id
            or self._tokenizer.convert_tokens_to_ids("</s>")
            or 2
        )

    def _run_inference(self, input_ids: np.ndarray) -> np.ndarray:
        """执行一次推理，返回 logits [batch, vocab_size]"""
        import tensorrt as trt
        import pycuda.driver as cuda

        input_shape = tuple(input_ids.shape)
        input_ids = input_ids.astype(np.int32)

        if self._use_v2_api:
            # TensorRT 8.x
            idx_in = self._engine.get_binding_index(self._input_name)
            idx_out = self._engine.get_binding_index(self._output_name)
            self._context.set_binding_shape(idx_in, input_shape)
            out_shape = tuple(self._context.get_binding_shape(idx_out))
            dtype = self._engine.get_binding_dtype(idx_out)
        else:
            # TensorRT 10.x
            self._context.set_input_shape(self._input_name, input_shape)
            out_shape = tuple(self._context.get_tensor_shape(self._output_name))
            dtype = self._context.get_tensor_dtype(self._output_name)

        dtype_map = {
            trt.DataType.FLOAT: np.float32,
            trt.DataType.HALF: np.float16,
            trt.DataType.INT32: np.int32,
        }
        np_dtype = dtype_map.get(dtype, np.float32)

        # 处理动态维度
        out_volume = 1
        for d in out_shape:
            out_volume *= (d if d > 0 else input_shape[-1])

        h_output = np.empty((out_volume,), dtype=np_dtype)
        d_input = cuda.mem_alloc(input_ids.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)

        cuda.memcpy_htod(d_input, input_ids)

        if self._use_v2_api:
            # bindings 顺序必须与 engine 的 binding 索引一致
            num_bindings = self._engine.num_bindings
            bindings = [0] * num_bindings
            bindings[idx_in] = int(d_input)
            bindings[idx_out] = int(d_output)
            self._context.execute_async_v2(bindings=bindings, stream_handle=int(self._stream))
        else:
            self._context.set_tensor_address(self._input_name, int(d_input))
            self._context.set_tensor_address(self._output_name, int(d_output))
            self._context.execute_async_v3(stream_handle=int(self._stream))

        cuda.memcpy_dtoh(h_output, d_output)
        self._stream.synchronize()

        # 取最后位置的 logits
        if len(out_shape) == 3:
            batch, seq, vocab = out_shape
            if batch == -1:
                batch = input_shape[0]
            if seq == -1:
                seq = input_shape[1]
            if vocab == -1:
                vocab = self._vocab_size or (out_volume // max(1, batch * seq))
            h_output = h_output.reshape(batch, seq, vocab)
            return h_output[:, -1, :]
        elif len(out_shape) == 2:
            return h_output.reshape(out_shape[0], -1)
        else:
            return h_output.reshape(-1, self._vocab_size or out_volume)

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
        if self._engine is None:
            raise RuntimeError("请先调用 load() 加载 engine")

        stop_ids = []
        if stop:
            for s in stop:
                ids = self._tokenizer.encode(s, add_special_tokens=False)
                if ids:
                    stop_ids.append(ids)

        results = []
        for prompt in prompts:
            output_ids = self._tokenizer.encode(prompt, add_special_tokens=True, return_tensors="np")
            if output_ids.ndim == 2:
                output_ids = output_ids[0]

            for _ in range(max_new_tokens):
                input_ids = output_ids.astype(np.int32)
                if input_ids.ndim == 1:
                    input_ids = input_ids.reshape(1, -1)

                logits = self._run_inference(input_ids)
                next_token = _sample_next_token(logits[0], temperature, top_k, top_p)
                output_ids = np.append(output_ids, next_token)

                if next_token == self._eos_id:
                    break
                if stop_ids:
                    max_len = max(len(s) for s in stop_ids)
                    tail = tuple(output_ids[-max_len:].tolist())
                    if any(tail[-len(s):] == tuple(s) for s in stop_ids):
                        break

            text = self._tokenizer.decode(output_ids, skip_special_tokens=True)
            if prompt and text.startswith(prompt):
                text = text[len(prompt):].strip()
            results.append(text)

        return results

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
        """流式生成"""
        if self._engine is None:
            raise RuntimeError("请先调用 load() 加载 engine")

        loop = asyncio.get_event_loop()

        def _sync_gen():
            return self.generate(
                [prompt],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
            )[0]

        text = await loop.run_in_executor(None, _sync_gen)
        for char in text:
            yield char

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._engine is not None
