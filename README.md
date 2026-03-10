# llm_run

基于原始 TensorRT 的 LLM 推理引擎，提供 OpenAI 兼容的 REST API 服务。
**适用于 Jetson Thor 等嵌入式环境**，不依赖 tensorrt_llm。

## 功能特性

- **原始 TensorRT 推理**：使用 tensorrt + pycuda 加载 .engine/.plan 并执行推理
- **Jetson 友好**：仅依赖 JetPack 自带的 tensorrt，适配嵌入式部署
- **OpenAI 兼容 API**：支持 `/v1/chat/completions`、`/v1/completions`、`/v1/models` 等接口
- **流式输出**：支持 SSE 流式返回
- **多模型参数**：temperature、top_p、top_k、stop 等

## 项目结构

```
llm_run/
├── llm_run/
│   ├── __init__.py
│   ├── main.py              # 主入口
│   ├── engine/              # 推理引擎模块
│   │   ├── base.py          # 抽象基类
│   │   └── tensorrt_engine.py  # TensorRT 实现
│   ├── api/                 # REST API 模块
│   │   ├── app.py           # FastAPI 应用
│   │   ├── routes.py        # 路由定义
│   │   ├── schemas.py       # 请求/响应模型
│   │   └── deps.py          # 依赖注入
│   └── config/              # 配置
│       ├── __init__.py
│       └── settings.py
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -e .

# Jetson 上：tensorrt、pycuda 通常随 JetPack 预装
# x86 开发机可额外安装：
# pip install tensorrt pycuda
```

## 使用

### 启动服务

```bash
# 指定 engine 和 tokenizer 路径（tokenizer 用于编码/解码）
llm-run --engine-path /path/to/your/engine.plan --tokenizer-path /path/to/tokenizer

# tokenizer-path 可为 HuggingFace 模型名或本地路径
llm-run \
  --engine-path ./engines/model.plan \
  --tokenizer-path Qwen/Qwen2-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

**Engine 格式说明**：需为单步推理 engine，输入 `input_ids` [batch, seq_len]，输出 `logits` [batch, seq, vocab] 或 [batch, vocab]。若 binding 名称不同，可在代码中通过 `input_name`/`output_name` 指定。

### API 调用示例

```bash
# Chat Completions
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llm",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 128
  }'

# 流式
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llm", "messages": [{"role": "user", "content": "你好"}], "stream": true}'

# 使用 OpenAI 客户端
# OPENAI_API_BASE=http://localhost:8000/v1 python -c "
# from openai import OpenAI
# client = OpenAI(base_url='http://localhost:8000/v1', api_key='dummy')
# r = client.chat.completions.create(model='llm', messages=[{'role':'user','content':'你好'}])
# print(r.choices[0].message.content)
# "
```

## 配置

| 环境变量 | 说明 |
|---------|------|
| `LLM_RUN_ENGINE_PATH` | TensorRT engine 路径 |
| `LLM_RUN_TOKENIZER_PATH` | Tokenizer 路径 |
| `LLM_RUN_HOST` | API 监听地址 |
| `LLM_RUN_PORT` | API 端口 |

## 开发

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
