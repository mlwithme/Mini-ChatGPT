from __future__ import annotations
import logging
from datetime import datetime  # 当前时间会被注入系统提示词中。
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.messages import convert_to_messages
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver
from .store import PostgresMemoryStore
from .prompts import SYSTEM_PROMPT
from .tools import build_postgres_memory_tool
from .context import Context
from .store import MEMORY_TYPE_LABELS, MemorySearchResult
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


def chat(state: MessagesState, runtime: Runtime[Context]) -> MessagesState:
    user_text = state["messages"][-1].content  # 取当前用户的输入
    memory_store = PostgresMemoryStore(runtime.store, runtime.context.memory_root)
    memory_hits = memory_store.search(runtime.context.user_id, user_text)  # 搜索历史记忆
    system_prompt = SYSTEM_PROMPT.format(
        user_info=_format_memories(memory_hits),
        time=datetime.now().isoformat(timespec="seconds"),
    )
    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        *state["messages"],
    ]
    tool = build_postgres_memory_tool(memory_store, runtime.context)
    runnable = load_chat_model(runtime.context.model).bind_tools([tool])

    for _ in range(5):
        ai_message = runnable.invoke(messages)
        messages.append(ai_message)
        logger.debug(f"本轮模型返回内容：{ai_message}")
        tool_calls = getattr(ai_message, "tool_calls", []) or []
        if not tool_calls:
            logger.debug(f"本轮未触发工具调用")
            return {"messages": [AIMessage(content=message_text(ai_message))]}
        logger.debug(f"本轮已触发工具调用，模型返回工具调用数量: count={len(tool_calls)}")
        for tool_call in tool_calls:
            logger.debug(f"模型请求调用工具: name={tool_call.get('name')}")
            logger.debug(f"工具调用参数: {tool_call}")
            result = tool.invoke(tool_call["args"])
            messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

    return {"messages": [AIMessage(content="本轮记忆工具调用次数过多，已停止继续执行。")]}


def get_agent(checkpointer: PostgresSaver, store: PostgresStore):
    agent = StateGraph(MessagesState, context_schema=Context)
    # context_schema=Context 的作用是告诉 LangGraph：
    # 这个 Graph 运行时会接收一个 Context 类型的运行时上下文，并且节点函数可以通过 runtime.context 访问它。
    # 例如上面 chat() 方法中的
    # runtime.context.user_id
    # runtime.context.thread_id
    # runtime.context.model
    # runtime.context.memory_root
    agent.add_node("chat", chat)
    agent.add_edge(START, "chat")
    agent.add_edge("chat", END)
    return agent.compile(checkpointer=checkpointer, store=store)


def invoke_agent(
        agent,
        user_id: str,
        thread_id: str,
        user_text: str,
        model: str,
        memory_root: str,
) -> str:
    input_data = {"messages": convert_to_messages([{"role": "user", "content": user_text}])}
    config = RunnableConfig({"configurable": {"thread_id": thread_id}})  # 这个 thread_id 是供 LangGraph 内部使用，例如保存短期记忆、快照等
    result = agent.invoke(
        input_data,
        config=config,
        context=Context(
            user_id=user_id,
            thread_id=thread_id,  # context 中的 thread_id 主要是供业务逻辑中使用
            model=model,
            memory_root=memory_root,
        ),
    )
    return result["messages"][-1].content
