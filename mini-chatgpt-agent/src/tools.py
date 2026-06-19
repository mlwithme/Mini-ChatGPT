from __future__ import annotations
import logging
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from .store import MemoryType, PostgresMemoryStore
from .context import Context

logger = logging.getLogger(__name__)

MEMORY_TOOL_DESCRIPTION = """
保存或更新用户的长期记忆。仅保存未来仍具有价值的信息，不要保存临时请求或当前会话的短期上下文。
memory_type 有3种情况:
- episodic：用户经历过的具体事件或经验。
- semantic：用户长期稳定的事实、偏好、身份、目标、背景或持续项目。
- procedural：用户希望助手长期遵循的工作方式或回答风格。

可以保存：
- 职业背景
- 技术栈
- 学习方向
- 长期目标
- 兴趣偏好
- 持续进行中的项目
- 回答风格偏好

不要保存：
- 一次性问题
- 临时任务
- 当前会话上下文
- 短期有效的信息

原则：只有当该信息未来仍可能帮助助手更好地服务用户时，才应保存。
""".strip()


class UpsertMemoryInput(BaseModel):
    content: str = Field(
        description="记忆的核心内容")

    context: str = Field(
        description="对这条记忆的补充上下文说明"
    )

    memory_type: MemoryType = Field(
        description="记忆类型：episodic（情景记忆）、semantic（语义记忆）或 procedural（程序记忆）"
    )

    memory_id: Optional[str] = Field(
        description="记忆的唯一标识，例如：0ec42eb7-3436-4cfb-9898-f6c69a80c5a2 。更新已有记忆时传入，新建记忆时为None"
    )


def build_postgres_memory_tool(
        store: PostgresMemoryStore,
        chat_context: Context,
) -> StructuredTool:
    logger.debug(f"构造记忆写入工具: user_id={chat_context.user_id}")

    def _upsert_memory(
            content: str,
            context: str,
            memory_type: MemoryType,
            memory_id: str | None,
    ) -> str:
        logger.debug(f"执行记忆写入工具: user_id={chat_context.user_id}, "
                     f"memory_id = {memory_id}, memory_type={memory_type}")
        record = store.upsert(
            user_id=chat_context.user_id,
            content=content,
            context=context,
            memory_type=memory_type,
            memory_id=memory_id,
            source_thread_id=chat_context.thread_id,
        )
        return f"记忆已经完成存储 {record.memory_id}"

    return StructuredTool.from_function(
        func=_upsert_memory,
        name="upsert_memory",
        description=MEMORY_TOOL_DESCRIPTION,
        args_schema=UpsertMemoryInput,
    )


__all__ = ["build_postgres_memory_tool", "MEMORY_TOOL_DESCRIPTION"]
