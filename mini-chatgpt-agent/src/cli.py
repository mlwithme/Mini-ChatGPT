from __future__ import annotations
import logging
from .store import MEMORY_TYPE_LABELS
from .store import PostgresMemoryStore
from .utils import make_thread_id
from .agent import invoke_agent
from langgraph.checkpoint.postgres import PostgresSaver

logger = logging.getLogger(__name__)


def _print_memories(memory_store: PostgresMemoryStore, user_id: str) -> None:
    memories = memory_store.list_memories(user_id)
    if not memories:
        logger.info(f"用户 {user_id} 暂无长期记忆")
        return

    logger.info("已保存记忆:")
    for index, memory in enumerate(memories, start=1):
        label = MEMORY_TYPE_LABELS.get(memory.memory_type, memory.memory_type)
        print(f"{index:02d}. [{memory.memory_type} / {label}] {memory.content} | {memory.context} | {memory.memory_id}")


def _print_welcome(args, memory_store: PostgresMemoryStore) -> None:
    """打印程序启动信息。"""
    # 打印程序标题。
    logger.info(f"\n{'=' * 10} Mini ChatGPT(Postgres存储长期记忆){'=' * 10}")
    # 打印当前 user_id。
    logger.info(f"用户ID = {args.user_id}")
    logger.info(f"会话ID = {args.thread_id}")
    logger.info(f"模型 = {args.model}")
    logger.info(f"记忆命名空间 = {memory_store.get_namespace(args.user_id)}")
    logger.info(f"已存储记忆数量 = {len(memory_store.list_memories(args.user_id))}")
    # 顺便显示帮助信息。
    _print_help()


def _print_help() -> None:
    """打印可用命令。"""
    # 命令尽量保持短小，方便终端中快速查看。
    logger.info(
        "Commands: /new 新会话, /thread <id> 切换会话, /memories: 查看当前用户所有长期记忆, \n/clear: 清空当前用户所有长期记忆, /clearall: 清空用户所有记忆, /exit: 退出, /help: 查看帮助")


def run_cli(
        agent,
        memory_store: PostgresMemoryStore,
        args,
        checkpointer: PostgresSaver
) -> None:
    logger.debug(f"服务启动中，正在加载环境")
    _print_welcome(args, memory_store)
    thread_id = args.thread_id
    while True:
        user_text = input("\n[You]: ").strip()
        if not user_text:
            continue
        if user_text == "/exit":
            logger.info("已退出 CLI")
            break
        if user_text == "/help":
            _print_help()
            continue
        if user_text == "/memories":
            logger.debug(f"查看用户记忆: user_id={args.user_id}")
            _print_memories(memory_store, args.user_id)
            continue
        if user_text == "/clear":
            logger.debug(f"清空用户长期记忆: user_id={args.user_id}")
            memory_store.clear_long(args.user_id)
            logger.info(f"用户 {args.user_id} 的所有长期记忆已清空，可通过命令 /memories 验证！")
            continue
        if user_text == "/clearall":
            logger.debug(f"清空用户所有记忆: user_id={args.user_id}")
            deleted_counts = memory_store.clearall(args.user_id, checkpointer, thread_id)
            logger.info(
                f"用户 {args.user_id} 的所有记忆已清空："
                f"长期记忆 {deleted_counts['long']} 条，短期记忆 {deleted_counts['short']} 个会话"
            )
            continue
        if user_text == "/new":
            thread_id = make_thread_id()
            logger.info(f"已新建会话：{thread_id}")
            continue
        if user_text.startswith("/thread "):
            thread_id = user_text.split(maxsplit=1)[1].strip()
            logger.info(f"已切换到会话：{thread_id}")
            continue
        if user_text == "/id":
            logger.info(f"当前会话ID: {thread_id}")
            continue
        logger.info(f"处理用户输入: {user_text}", extra={"user_input": True})
        answer = invoke_agent(agent, thread_id, user_text,args)
        logger.info(f"[Assistant]:\n {answer}")
        logger.info('-' * 70)


# 导出 CLI 启动函数。
__all__ = ["run_cli"]
