from __future__ import annotations

import argparse
import os
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver
from src.logging_config import configure_logging
from src.qwen import get_embeddings_model
from src.store import PostgresMemoryStore
from src.cli import run_cli
from src.agent import get_agent
from src.streamlit import STREAMLIT_AVAILABLE, run_streamlit_app
from src.utils import make_thread_id
import logging

logger = logging.getLogger(__name__)

DB_URI = os.getenv("DB_URI", "postgresql://db_name:your_password@your_host_ip:5432/mypg?sslmode=disable")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "your_dashscope_api_key")


def get_memory_index() -> dict | None:
    """为 PostgresStore 启用 pgvector 语义检索索引。"""
    if not os.getenv("DASHSCOPE_API_KEY"):
        return None

    return {
        "embed": get_embeddings_model(),
        "dims": 1024,
        "fields": ["content", "context"],
    }


def build_parser() -> argparse:
    thread_id = make_thread_id()
    parser = argparse.ArgumentParser(description="LangGraph ChatGPT demo with PostgreSQL memory")
    parser.add_argument("--cli", action="store_true", help="使用命令行模式，而不是 Streamlit 页面")
    parser.add_argument("--user-id", default='demo-user')
    parser.add_argument("--thread-id", default=thread_id)
    # parser.add_argument("--model", default='qwen-plus')
    parser.add_argument("--model", default='qwen3.7-plus')
    parser.add_argument("--memory_root", default='demo_memory')
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志输出等级",
    )
    # 日志默认保存到本地文件，也允许用户指定其他路径。
    parser.add_argument(
        "--log-file",
        default="logs/agent.log",
        help="日志保存目录",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    memory_index = get_memory_index()
    configure_logging(args.log_level, args.log_file)
    with (
        PostgresStore.from_conn_string(DB_URI, index=memory_index) as store,
        PostgresSaver.from_conn_string(DB_URI) as checkpointer,
    ):
        store.setup()
        checkpointer.setup()
        agent = get_agent(checkpointer, store)
        memory_store = PostgresMemoryStore(store, args.memory_root)
        # print(agent.get_graph().draw_mermaid()) # 输出图结构
        if args.cli or not STREAMLIT_AVAILABLE:
            run_cli(agent, memory_store, args, checkpointer)
        else:
            run_streamlit_app(agent, memory_store, args, checkpointer)


if __name__ == "__main__":
    main()

# streamlit run main.py
# python main.py --cli
