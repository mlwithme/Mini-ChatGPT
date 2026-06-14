from __future__ import annotations
import logging
from dataclasses import dataclass  # dataclass 用于组织一轮对话的返回结果。
from datetime import datetime  # 当前时间会被注入系统提示词中。
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from .context import Context
from .store import MEMORY_TYPE_LABELS, JsonMemoryStore, MemorySearchResult
from .tools import build_upsert_memory_tool
from .utils import load_chat_model

logger = logging.getLogger(__name__)


# 这个 dataclass 表示“一轮对话”的最终产物。
@dataclass
class AgentReply:
    """表示 Agent 单轮响应的结果。"""

    # 最终返回给用户的文本。
    text: str
    # 更新后的历史消息（短期记忆），下一轮还会继续用到。
    history: list[BaseMessage]
    # 本轮参与检索并注入给模型的记忆列表。
    used_memories: list[MemorySearchResult]


class MemoryAgent:
    """一个本地优先、带长期记忆的小型聊天助手。"""

    def __init__(self, context: Context, store: JsonMemoryStore):
        """初始化 Agent。"""
        # 保存上下文配置。
        self.context = context
        # 保存记忆存储对象。
        self.store = store
        logger.debug(f"初始化 MemoryAgent: user_id={context.user_id}, model={context.model}")
        # 根据配置加载聊天模型。
        self.llm = load_chat_model(context.model)  # 这里默认使用 qwen
        # 为当前用户构造一个“保存记忆”工具。
        self.memory_tool = build_upsert_memory_tool(store, context.user_id)

    def respond(
            self,
            user_message: str,
            *,
            history: list[BaseMessage] | None = None,
            memory_limit: int = 5,
            max_loops: int = 5,
    ) -> AgentReply:
        """
        处理一条用户消息，并返回回答结果。
        :param user_message:  用户当前的输入内容
        :param history: 交互过程中的会话（短期）记忆，初始时为空列表
        :param memory_limit: 检索用户长期记忆时的最大返回条数
        :param max_loops:
        :return:
        """
        # 复制历史消息，避免直接修改调用者传入的原列表。
        dialogue_history = list(history or [])
        logger.debug(
            f"开始处理消息: user_id={self.context.user_id},"
            f" history_messages={len(dialogue_history)}"
        )
        # 先根据当前用户输入检索相关旧记忆。
        memory_hits = self.store.search(
            self.context.user_id,
            query=user_message,
            limit=memory_limit,
        )
        logger.debug(f"检索到相关记忆: count={len(memory_hits)}", )
        # 把检索结果和当前时间填入系统提示词模板。
        system_prompt = self.context.system_prompt.format(
            user_info=self._format_memories(memory_hits),
            time=datetime.now().isoformat(timespec="seconds"),  # 时间精确到秒
        )
        # 组装要发送给模型的完整消息列表。
        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),  # 原系统提示词 + 用户历史记忆 + 当前时间
            *dialogue_history,  # 短期记忆
            HumanMessage(content=user_message),
        ]
        # 绑定工具，让模型可以决定是否保存新记忆。
        runnable = self.llm.bind_tools([self.memory_tool])

        # 某些模型可能会经历“工具调用 -> 再回答”的多轮闭环，
        # 因此这里允许最多循环若干次。
        for i in range(max_loops):
            # 请求模型生成下一步动作。
            logger.debug(f"本轮模型开始请求第{i + 1}/{max_loops}次")
            ai_message = runnable.invoke(messages)
            # 把模型的输出加入当前消息流。
            messages.append(ai_message)
            logger.debug(f"本轮模型返回内容：{ai_message}")
            # 读取模型这一步发出的工具调用请求。
            tool_calls = getattr(ai_message, "tool_calls", []) or []
            if tool_calls:
                logger.debug(f"本轮已触发工具调用，模型返回工具调用数量: count={len(tool_calls)}")
            else:
                logger.debug(f"本轮未触发工具调用")
            # 如果没有工具调用，说明模型已经给出了最终回答。
            if not tool_calls:
                # 为下一轮准备新的历史消息。
                updated_history = [
                    *dialogue_history,
                    HumanMessage(content=user_message),
                    AIMessage(content=self._message_text(ai_message)),
                ]
                # 返回最终结果。
                logger.debug(f"模型完成最终回复: loops={i + 1}")
                return AgentReply(
                    text=self._message_text(ai_message),
                    history=updated_history,
                    used_memories=memory_hits,
                )

            # 如果模型请求了工具，就逐个执行。
            logger.info(f"有 {len(tool_calls)} 条记忆插入")
            for tool_call in tool_calls:
                # 调用保存记忆工具，并传入模型给出的参数。
                logger.debug(f"模型请求调用工具: name={tool_call.get('name')}")
                logger.debug(f"工具调用参数: {tool_call}")
                result = self.memory_tool.invoke(tool_call["args"])
                # 把工具执行结果包装成 ToolMessage 再喂回模型。
                messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

        # 如果循环很多次仍没有生成最终答案，就抛出异常提醒调用者。
        logger.error(f"模型持续调用工具，未能完成最终回复: max_loops={max_loops}")
        raise RuntimeError("已达到本轮交互最大次数")

    @staticmethod
    def _message_text(message: AIMessage) -> str:
        """把模型消息提取成最终可显示的文本。"""
        # 最常见情况：模型内容本身就是一个字符串。
        if isinstance(message.content, str):
            return message.content.strip()
        # 某些模型会返回结构化内容块列表。
        if isinstance(message.content, list):
            # 我们只提取其中的文本块。
            chunks = []
            for item in message.content:
                # 只处理 `type == "text"` 的块，其余类型忽略。
                if isinstance(item, dict) and item.get("type") == "text":
                    chunks.append(item.get("text", ""))
            # 把多段文本合并成一个字符串。
            return "\n".join(part for part in chunks if part).strip()
        # 兜底处理：把未知内容类型强制转成字符串。
        return str(message.content).strip()

    @staticmethod
    def _format_memories(memory_hits: list[MemorySearchResult]) -> str:
        """把检索出的记忆格式化成提示词片段。"""
        # 如果还没有任何旧记忆，就明确告诉模型。
        if not memory_hits:
            return "没有发现已存储的长期记忆"
        # 把每条命中记忆格式化成一行可读文本。
        lines = []
        for item in memory_hits:
            label = MEMORY_TYPE_LABELS.get(item.record.memory_type, item.record.memory_type)
            lines.append(
                f"- [{item.record.memory_type} / {label} / memory_id = {item.record.memory_id}] "
                f"{item.record.content} | context: {item.record.context} | score: {item.score:.3f}"
            )
        return "\n".join(lines)


def create_agent(context: Context) -> MemoryAgent:
    """根据运行配置构造一个完整可用的本地记忆助手。"""
    # 延迟导入，避免仅查看模块时就触发模型相关初始化。
    from .qwen import get_embeddings_model

    # 创建本地记忆存储，并挂上向量模型用于语义检索。
    logger.debug(f"创建本地记忆存储: memory_dir={context.memory_dir}")
    store = JsonMemoryStore(context.memory_dir, embeddings_model=get_embeddings_model())
    # 返回一个配置完成的 Agent 实例。
    return MemoryAgent(context=context, store=store)


# 模块公开导出。
__all__ = ["AgentReply", "MemoryAgent", "create_agent"]
