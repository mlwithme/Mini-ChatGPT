from __future__ import annotations

import logging
from .agent import create_agent
from .context import Context
from .store import MEMORY_TYPE_LABELS, JsonMemoryStore

logger = logging.getLogger(__name__)


def run_cli(args) -> None:
    """启动命令行版本地记忆助手。"""

    # 根据参数初始化日志。

    logger.debug(f"服务启动中，正在加载环境")
    # 把参数组装成统一的上下文配置对象。
    context = Context(
        user_id=args.user_id,
        model=args.model,
        memory_dir=args.memory_dir,
    )
    # 创建 Agent 实例，同时也会拿到它内部使用的存储对象。
    agent = create_agent(context)
    store = agent.store

    # 如果指定了 `--once`，就执行一轮问答后直接退出。
    if args.once:
        logger.debug("执行单轮问答模式")
        reply = agent.respond(args.once)
        logger.info(reply.text)
        return

    # 否则先打印欢迎信息，再进入交互循环。
    _print_welcome(context, store)
    # 这个列表用来保存当前终端会话内的短期上下文。
    history = []

    # 一直循环，直到用户主动退出。
    while True:
        # 读取用户输入。
        user_text = input("\n[You]: ").strip()
        # 空输入直接忽略。
        if not user_text:
            continue
        # `/exit`：退出程序。
        if user_text == "/exit":
            logger.info("已退出 CLI")
            break
        # `/help`：打印帮助。
        if user_text == "/help":
            _print_help()
            continue
        # `/memories`：查看当前用户已保存记忆。
        if user_text == "/memories":
            logger.debug(f"查看用户记忆: user_id={context.user_id}")
            _print_memories(store, context.user_id)
            continue
        # `/clear`：清空当前用户的记忆文件。
        if user_text == "/clear":
            logger.debug(f"清空用户记忆: user_id={context.user_id}")
            store.clear(context.user_id)
            logger.info(f"用户 {context.user_id} 的所有长期记忆已清空，可通过命令 /memories 验证！")
            continue

        # 普通文本则交给 Agent 处理。
        logger.info(f"处理用户输入: {user_text}", extra={"user_input": True})
        reply = agent.respond(user_text, history=history)
        # 保存更新后的历史消息，供下一轮继续使用。
        history = reply.history
        # 把回答输出到终端。
        logger.info(f"[Assistant]:\n {reply.text}")
        logger.info('-' * 70)


def _print_welcome(context: Context, store: JsonMemoryStore) -> None:
    """打印程序启动信息。"""
    # 打印程序标题。
    logger.info(f"\n{'=' * 10} Mini ChatGPT(本地存储长期记忆){'=' * 10}")
    # 打印当前 user_id。
    logger.info(f"用户ID = {context.user_id}")
    # 打印当前使用的模型。
    logger.info(f"模型 = {context.model}")
    # 打印记忆文件目录。
    logger.info(f"记忆目录 = {context.memory_dir}")
    # 打印当前用户已有多少条记忆。
    logger.info(f"已存储记忆数量 = {len(store.list_memories(context.user_id))}")
    # 顺便显示帮助信息。
    _print_help()


def _print_help() -> None:
    """打印可用命令。"""
    # 命令尽量保持短小，方便终端中快速查看。
    logger.info(
        "Commands: /help: 查看帮助  /memories: 查看当前用户所有长期记忆  /clear: 清楚当前用户所有长期记忆 /exit: 退出")


def _print_memories(store: JsonMemoryStore, user_id: str) -> None:
    """打印某个用户的全部记忆。"""
    # 读取该用户的所有记忆。
    memories = store.list_memories(user_id)
    # 如果为空，就给出提示。
    if not memories:
        logger.info(f"用户 {user_id} 暂无长期记忆")
        return
    # 打印小标题。
    logger.info("已保存记忆:")
    # 把每条记忆按编号打印出来。
    for index, memory in enumerate(memories, start=1):
        label = MEMORY_TYPE_LABELS.get(memory.memory_type, memory.memory_type)
        logger.info(f"{index}. [{memory.memory_type} / {label}] {memory.content} | {memory.context}")


# 导出 CLI 启动函数。
__all__ = ["run_cli"]
