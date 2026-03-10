"""
llm_run 主入口
"""

import argparse
import uvicorn

from llm_run.api import create_app


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Run - TensorRT 推理引擎与 OpenAI 兼容 API")
    parser.add_argument(
        "--engine-path",
        type=str,
        required=True,
        help="TensorRT engine 文件或目录路径",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Tokenizer 路径（默认与 engine 同目录）",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="API 监听地址",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API 端口",
    )
    parser.add_argument(
        "--api-prefix",
        type=str,
        default="/v1",
        help="API 路径前缀",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = create_app(
        engine_path=args.engine_path,
        tokenizer_path=args.tokenizer_path,
        api_prefix=args.api_prefix,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
