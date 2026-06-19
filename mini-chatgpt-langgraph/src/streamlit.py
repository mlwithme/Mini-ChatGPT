from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver

from .agent import invoke_agent
from .store import MEMORY_TYPE_LABELS, PostgresMemoryStore
from .utils import make_thread_id

try:
    import streamlit as st
except ImportError:
    st = None

STREAMLIT_AVAILABLE = st is not None


def _get_query_thread_id() -> str | None:
    """从页面 URL 查询参数中读取当前会话 ID，用于刷新页面后恢复同一个 thread。"""
    if hasattr(st, "query_params"):
        return st.query_params.get("thread_id")
    values = st.experimental_get_query_params().get("thread_id")
    return values[0] if values else None


def _set_query_thread_id(thread_id: str) -> None:
    """把当前会话 ID 写入页面 URL 查询参数，让浏览器刷新后仍能定位到当前 thread。"""
    if hasattr(st, "query_params"):
        st.query_params["thread_id"] = thread_id
        return
    st.experimental_set_query_params(thread_id=thread_id)


def _message_content(message: BaseMessage) -> str:
    """把 LangChain 消息对象中的 content 统一转换为 Streamlit 可以展示的文本。"""
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        chunks = []
        for item in message.content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(item.get("text", ""))
            else:
                chunks.append(str(item))
        return "\n".join(chunk for chunk in chunks if chunk)
    return str(message.content)


def _message_role(message: BaseMessage) -> str | None:
    """把 LangChain 消息类型映射为 Streamlit chat_message 需要的 role。"""
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    return None


def _load_messages_from_checkpoint(agent, thread_id: str) -> list[dict[str, str]]:
    """根据 thread_id 从 LangGraph checkpointer 中读取短期记忆，并转换为页面聊天记录。"""
    config = RunnableConfig({"configurable": {"thread_id": thread_id}})
    state = agent.get_state(config)
    messages = state.values.get("messages", []) if state and state.values else []
    loaded_messages = []
    for message in messages:
        role = _message_role(message)
        if role:
            loaded_messages.append({"role": role, "content": _message_content(message)})
    return loaded_messages


def ensure_session_state(agent, args) -> None:
    """初始化 Streamlit 页面状态，并在需要时从 LangGraph checkpoint 恢复当前会话消息。"""
    thread_id = _get_query_thread_id() or args.thread_id
    _set_query_thread_id(thread_id)
    st.session_state.setdefault("user_id", args.user_id)
    st.session_state.setdefault("threads", [thread_id])
    st.session_state.setdefault("thread_id", thread_id)
    st.session_state.setdefault("messages_by_thread", {})
    if thread_id not in st.session_state.threads:
        st.session_state.threads.insert(0, thread_id)
    if thread_id not in st.session_state.messages_by_thread:
        st.session_state.messages_by_thread[thread_id] = _load_messages_from_checkpoint(agent, thread_id)
    # 在 Streamlit 的会话状态里初始化一个“按 thread_id 分组保存前端聊天记录”的字典。


def _remove_current_thread_from_page() -> str:
    """从页面会话列表和聊天展示缓存中移除当前会话，并切换到可用的新会话。"""
    current_thread_id = st.session_state.thread_id
    current_messages = st.session_state.messages_by_thread.get(current_thread_id, [])
    if not current_messages:
        _set_query_thread_id(current_thread_id)
        return current_thread_id

    st.session_state.messages_by_thread.pop(current_thread_id, None)
    st.session_state.threads = [
        thread_id for thread_id in st.session_state.threads
        if thread_id != current_thread_id
    ]
    if not st.session_state.threads:
        new_thread_id = make_thread_id()
        st.session_state.threads = [new_thread_id]
        st.session_state.messages_by_thread[new_thread_id] = []
    st.session_state.thread_id = st.session_state.threads[0]
    _set_query_thread_id(st.session_state.thread_id)
    return current_thread_id


def run_streamlit_app(
        agent,
        memory_store: PostgresMemoryStore,
        args,
        checkpointer: PostgresSaver,
) -> None:
    st.set_page_config(page_title="Mini-ChatGPT", layout="wide")
    ensure_session_state(agent, args)

    with st.sidebar:
        st.title("Mini-ChatGPT")
        if st.button("新建对话", use_container_width=True):
            thread_id = make_thread_id()
            st.session_state.threads.insert(0, thread_id)
            st.session_state.thread_id = thread_id
            st.session_state.messages_by_thread[thread_id] = []
            _set_query_thread_id(thread_id)

        st.session_state.thread_id = st.radio(
            "历史对话",
            st.session_state.threads,
            index=st.session_state.threads.index(st.session_state.thread_id),
        )
        _set_query_thread_id(st.session_state.thread_id)

        with st.expander("记忆操作", expanded=False):
            if st.button("删除长期记忆", use_container_width=True):
                deleted_count = memory_store.clear_long(st.session_state.user_id)
                st.success(f"已删除 {deleted_count} 条长期记忆")

            if st.button("删除短期记忆", use_container_width=True):
                deleted_count = memory_store.clear_short(
                    st.session_state.user_id,
                    checkpointer,
                    st.session_state.thread_id,
                )
                deleted_thread_id = _remove_current_thread_from_page()
                st.success(f"已删除 {deleted_count} 个短期记忆会话：{deleted_thread_id}")

            if st.button("删除所有记忆", use_container_width=True):
                deleted_counts = memory_store.clearall(
                    st.session_state.user_id,
                    checkpointer,
                    st.session_state.thread_id,
                )
                deleted_thread_id = _remove_current_thread_from_page()
                st.success(
                    f"已删除 {deleted_counts['long']} 条长期记忆，"
                    f"并删除 {deleted_counts['short']} 个短期记忆会话：{deleted_thread_id}"
                )

        with st.expander("长期记忆", expanded=False):
            memories = memory_store.list_memories(st.session_state.user_id)
            if not memories:
                st.caption("暂无长期记忆")
            for memory in memories:
                label = MEMORY_TYPE_LABELS.get(memory.memory_type, memory.memory_type)
                st.markdown(f"- **{label}**：{memory.content}")

    st.title("Chat")
    thread_id = st.session_state.thread_id
    messages = st.session_state.messages_by_thread.setdefault(thread_id, [])

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("发消息给助手"):
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                answer = invoke_agent(
                    agent,
                    st.session_state.user_id,
                    thread_id,
                    prompt,
                    args.model,
                    args.memory_root,
                )
            st.markdown(answer)
        messages.append({"role": "assistant", "content": answer})
        st.rerun()
