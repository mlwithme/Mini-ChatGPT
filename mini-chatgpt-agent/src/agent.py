from __future__ import annotations

import logging
from datetime import datetime
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

from .context import Context
from .prompts import SYSTEM_PROMPT
from .store import MEMORY_TYPE_LABELS, MemorySearchResult, PostgresMemoryStore
from .tools import build_postgres_memory_tool
from .utils import load_chat_model

logger = logging.getLogger(__name__)


def _format_memories(memory_hits: list[MemorySearchResult]) -> str:
    if not memory_hits:
        return "没有发现已存储的长期记忆"

    lines = []
    for item in memory_hits:
        label = MEMORY_TYPE_LABELS.get(item.record.memory_type, item.record.memory_type)
        lines.append(
            f"- [{item.record.memory_type} / {label} / memory_id={item.record.memory_id}] "
            f"{item.record.content} | context: {item.record.context}"
        )
    return "\n".join(lines)


def message_text(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content.strip()
    if isinstance(message.content, list):
        chunks = []
        for item in message.content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(item.get("text", ""))
        return "\n".join(part for part in chunks if part).strip()
    return str(message.content).strip()


class CreateAgentApp:
    """使用 langchain.agents.create_agent 封装的聊天助手。"""

    def __init__(self, checkpointer: PostgresSaver, store: PostgresStore):
        self.checkpointer = checkpointer
        self.store = store

    def get_memory_store(self, memory_root: str) -> PostgresMemoryStore:
        return PostgresMemoryStore(self.store, memory_root)

    def build_runtime_agent(self, *, context, user_text: str,
                            search_memory_limit: int, recursion_limit: int = 20):
        def build_system_prompt(
                memory_store: PostgresMemoryStore,
                user_id: str,
                user_text: str,
                search_memory_limit: int, ) -> str:
            memory_hits = memory_store.search(user_id, user_text, search_memory_limit)
            return SYSTEM_PROMPT.format(user_info=_format_memories(memory_hits),
                                        time=datetime.now().isoformat(timespec="seconds"))

        system_prompt = build_system_prompt(context.memory_store, context.user_id, user_text, search_memory_limit)
        tool = build_postgres_memory_tool(context.memory_store, context)
        return create_agent(load_chat_model(context.model),
                            [tool],
                            system_prompt=system_prompt,
                            checkpointer=self.checkpointer,
                            store=self.store,
                            context_schema=Context).with_config(recursion_limit=recursion_limit)

    def get_state(self, *, user_id: str, thread_id: str,
                  model: str, memory_root: str, search_memory_limit: int = 10):
        """
        :param user_id:
        :param thread_id:
        :param model:
        :param memory_root:
        :param search_memory_limit:
        :return:
        """
        state_agent = create_agent(load_chat_model(model),checkpointer=self.checkpointer)
        # 只是借这个 agent 对象去访问同一个 checkpointer 里的状态，不需要其它任何东西
        config = RunnableConfig(
            {
                "configurable": {
                    "thread_id": thread_id,
                    "user_id": user_id,
                    "memory_root": memory_root,
                    "search_memory_limit": search_memory_limit,
                }
            }
        )
        return state_agent.get_state(config)


def get_agent(checkpointer: PostgresSaver, store: PostgresStore) -> CreateAgentApp:
    return CreateAgentApp(checkpointer=checkpointer, store=store)


def invoke_agent(
        agent: CreateAgentApp,
        thread_id: str,
        user_text: str,
        args,
) -> str:
    memory_store = agent.get_memory_store(args.memory_root)
    context = Context(user_id=args.user_id, thread_id=thread_id, model=args.model,
                      memory_root=args.memory_root, memory_store=memory_store)
    config = RunnableConfig({"configurable": {"thread_id": thread_id, "user_id": context.user_id,
                                              "memory_root": context.memory_root,
                                              "search_memory_limit": args.search_memory_limit}})
    runtime_agent = agent.build_runtime_agent(context=context, user_text=user_text,
                                              search_memory_limit=args.search_memory_limit,
                                              recursion_limit=args.recursion_limit)
    last_msg = ""
    for i, event in enumerate(runtime_agent.stream({"messages": [{"role": "user", "content": user_text}]},
                                                   stream_mode="values",
                                                   # values 表示每次全量输出完整消息，updates 每次输出增量，messages 返回 Token 流。
                                                   config=config)):
        logger.debug(f"\n\n{'*' * 10}")
        logger.debug(f"第 {i + 1}/{args.recursion_limit} 次循环\n"
                     f"当前短期记忆内容: {event}")
        logger.debug(f"\n{'*' * 10}\n")
        last_msg = message_text(event["messages"][-1])
    return last_msg


__all__ = ["CreateAgentApp", "get_agent", "invoke_agent"]
