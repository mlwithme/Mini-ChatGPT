from __future__ import annotations
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver

logger = logging.getLogger(__name__)

MemoryType = Literal["episodic", "semantic", "procedural"]

MEMORY_TYPE_LABELS: dict[str, str] = {
    "episodic": "情景记忆",
    "semantic": "语义记忆",
    "procedural": "程序记忆",
}


# 一条记忆记录对应一个 dataclass。
@dataclass
class MemoryRecord:
    # 记忆的唯一标识。
    memory_id: str
    # 记忆的核心内容。
    content: str
    # 对这条记忆的补充上下文说明。
    context: str
    # 首次创建时间。
    created_at: str
    # 最近一次更新时间。
    updated_at: str
    # 记忆类型：情景记忆、语义记忆或程序性记忆。
    memory_type: MemoryType = "semantic"

    # # 可选的向量表示，用于语义检索。LangGraph 中不需要
    # embedding: list[float] | None = None

    @property
    def text(self) -> str:
        """把内容和上下文拼成一段统一文本。"""
        # 便于后续检索或回退到关键词匹配。
        return f"{self.memory_type}\n{self.content}\n{self.context}".strip()


# 这个对象表示“检索结果 = 记忆 + 分数”。
@dataclass
class MemorySearchResult:
    """表示一条命中的记忆检索结果， 返回该条结果以及与用户请求之间的相似度"""
    # 命中的记忆记录本体。
    record: MemoryRecord
    # 用于排序的相似度分数。
    score: float


class PostgresMemoryStore:

    def __init__(self,
                 store: PostgresStore,
                 memory_root: str):
        self.store = store
        self.memory_root = memory_root
        logger.debug(f"初始化 PostgresMemoryStore 记忆存储: memory_root={self.memory_root}")

    def get_namespace(self, user_id: str) -> tuple[str, str]:
        """
        根据 user_id 返回命名空间，memory_root 是一个类似固定的根目录，由根目录+用户id 构成以为的命名空间
        :param user_id:
        :return:
        """
        return self.memory_root, user_id

    def list_memories(self, user_id: str) -> list[MemoryRecord]:
        """
        返回用户 user_id 的所有长期记忆
        :param user_id:
        :return:
        """
        namespace = self.get_namespace(user_id)
        records = []
        offset = 0
        page_size = 100

        while True:
            items = self.store.search(namespace, limit=page_size, offset=offset)
            if not items:
                break

            for item in items:
                record = self._item_to_record(item)
                if record:
                    records.append(record)

            if len(items) < page_size:
                break
            offset += page_size

        records.sort(key=lambda record: record.updated_at, reverse=True)
        logger.debug(f"读取用户记忆: user_id={user_id}, count={len(records)}")
        return records

    def search(self, user_id: str, query: str, limit: int = 12, ) -> list[MemorySearchResult]:
        results: list[MemorySearchResult] = []
        namespace = self.get_namespace(user_id)  # 得到命名空间
        items = self._search(namespace, query, limit)
        if not items:
            logger.debug(f"没有可检索的记忆: user_id={user_id}")
            return results
        # 返回与 query 最相关的 limit 条记忆内容
        logger.debug(f"开始在数据库中查询 user_id={user_id} 的记忆内容，query = {query}")
        # items: 我们自己定义的内容是存放在 value 这个字段里面的，是一个字典
        # [Item(namespace=['demo_memory_root1', 'demo-user2'],
        # key='38e1152c-12ff-4346-8a81-c59c2f51fe96',
        # value={'source': 'chat', 'content': '用户有一位名叫张三的2岁朋友，籍贯四川', 'context': '用户提及张三为‘2的朋友’，结合中文常见表达习惯及语境（如‘我有一个2岁的朋友’），此处‘2’应为年龄笔误或简写，合理推断为‘2岁’；该信息属于用户社交关系中的具体人物事实，具有长期识别与后续对话参考价值', 'memory_id': '38e1152c-12ff-4346-8a81-c59c2f51fe96', 'created_at': '2026-06-11T19:13:42', 'updated_at': '2026-06-11T19:13:42', 'memory_type': 'episodic', 'source_thread_id': '7a09aebf-9ee3-40e7-95bc-8a2a4513b635'},
        # created_at='2026-06-11T19:13:43.729072+08:00',
        # updated_at='2026-06-11T19:13:43.729072+08:00',
        # score=0.6269986442625952),
        # Item(namespace=['demo_memory_root1', 'demo-user2'], key='3e94fa30-63d0-452d-8b33-ba8b6dd9e099', value={'source': 'chat', 'content': '用户是上海人，计算机专业，刚大学毕业', 'context': '用户身份背景信息：籍贯、教育阶段、专业领域，属于长期稳定的个人事实，影响后续职业发展、技术交流或本地化建议等场景', 'memory_id': '3e94fa30-63d0-452d-8b33-ba8b6dd9e099', 'created_at': '2026-06-11T19:10:30', 'updated_at': '2026-06-11T19:10:30', 'memory_type': 'semantic', 'source_thread_id': '7a09aebf-9ee3-40e7-95bc-8a2a4513b635'}, created_at='2026-06-11T19:10:31.237873+08:00', updated_at='2026-06-11T19:10:31.237873+08:00', score=0.6184553135362092)]
        for item in items:
            record = self._item_to_record(item)
            # 将每一条记忆规整化为 MemoryRecord 格式
            if record:
                results.append(
                    MemorySearchResult(
                        record=record,
                        score=float(getattr(item, "score", 0.0) or 0.0),
                    )
                )

        results.sort(
            key=lambda item: (item.score, item.record.updated_at),
            reverse=True,
        )  # 按（分数高到低-更新时间降序） 排序
        logger.debug(f"完成记忆检索: total={len(results)}, returned={len(results)}")
        # results：
        # [MemorySearchResult(record=MemoryRecord(memory_id='38e1152c-12ff-4346-8a81-c59c2f51fe96', content='用户有一位名叫张三的2岁朋友，籍贯四川', context='用户提及张三为‘2的朋友’，结合中文常见表达习惯及语境（如‘我有一个2岁的朋友’），此处‘2’应为年龄笔误或简写，合理推断为‘2岁’；该信息属于用户社交关系中的具体人物事实，具有长期识别与后续对话参考价值', created_at='2026-06-11T19:13:42', updated_at='2026-06-11T19:13:42', memory_type='episodic'), score=0.6269986442625952),
        # MemorySearchResult(record=MemoryRecord(memory_id='3e94fa30-63d0-452d-8b33-ba8b6dd9e099', content='用户是上海人，计算机专业，刚大学毕业', context='用户身份背景信息：籍贯、教育阶段、专业领域，属于长期稳定的个人事实，影响后续职业发展、技术交流或本地化建议等场景', created_at='2026-06-11T19:10:30', updated_at='2026-06-11T19:10:30', memory_type='semantic'), score=0.6184553135362092)]
        return results

    def upsert(
            self,
            user_id: str,
            content: str,
            context: str,
            *,
            memory_type: MemoryType = "semantic",
            memory_id: str | None = None,
            source_thread_id: str | None = None,
    ) -> MemoryRecord:
        """
        插入新记忆，或更新已经存在的长期记忆
        :param user_id:
        :param content:
        :param context:
        :param memory_type:
        :param memory_id: 不为 None 则表示是已存在的记忆，None 则是新记忆内容
        :param source_thread_id:
        :return:
        """
        normalized_memory_type = self._normalize_memory_type(memory_type)
        namespace = self.get_namespace(user_id)
        content = content.strip()
        context = context.strip()
        now = datetime.now().isoformat(timespec="seconds")

        existing = self._find_existing(namespace, content, memory_id)
        # 根据传入的 namespace 和 memory_id 去数据库里取对应的长期记忆
        # 如果返回的 existing 为 None，则表示是新的记忆内容
        record_id = existing.memory_id if existing else str(uuid.uuid4())
        created_at = existing.created_at if existing else now

        value = {
            "memory_id": record_id,
            "memory_type": normalized_memory_type,
            "content": content,
            "context": context,
            "source": "chat",
            "source_thread_id": source_thread_id,
            "created_at": created_at,
            "updated_at": now,
        }
        self.store.put(namespace, record_id, value)  # 更新/新插入 长期记忆
        if not existing:
            logger.debug(f"新增记忆: user_id={user_id}, memory_id={record_id}, "
                         f"memory_type={normalized_memory_type}")
        else:
            logger.debug(f"更新记忆: user_id={user_id}, memory_id={record_id}, "
                         f"memory_type={normalized_memory_type}")
        return MemoryRecord(
            memory_id=record_id,
            content=content,
            context=context,
            created_at=created_at,
            updated_at=now,
            memory_type=normalized_memory_type,
        )

    def clear_long(self, user_id: str) -> int:
        """
        清空当前用户的所有长期记忆。
        :param user_id:
        :return: 被删除的记忆数量
        """
        namespace = self.get_namespace(user_id)
        deleted_count = 0

        while True:
            items = self.store.search(namespace, limit=100)
            if not items:
                break

            for item in items:
                self.store.delete(namespace, item.key)
                deleted_count += 1

        logger.debug(f"已清空用户长期记忆: user_id={user_id}, count={deleted_count}")
        return deleted_count

    @staticmethod
    def clear_short(
            user_id: str,
            checkpointer: PostgresSaver,
            thread_id: str,
    ) -> int:
        """
        清空指定会话的短期记忆。
        :param user_id:
        :param checkpointer:
        :param thread_id: 需要删除短期记忆的会话 ID
        :return: 被删除的短期记忆会话数量
        """
        checkpointer.delete_thread(thread_id)
        logger.debug(f"已清空用户短期记忆: user_id={user_id}, thread_id={thread_id}")
        return 1

    def clearall(
            self,
            user_id: str,
            checkpointer: PostgresSaver,
            thread_id: str,
    ) -> dict[str, int]:
        """
        清空当前用户的所有长期记忆，以及指定会话的短期记忆。
        :param user_id:
        :param checkpointer:
        :param thread_id: 需要删除短期记忆的会话 ID
        :return: 被删除的长期记忆数量和短期记忆会话数量
        """
        short_deleted_count = self.clear_short(user_id, checkpointer, thread_id)
        long_deleted_count = self.clear_long(user_id)
        logger.debug(
            f"已清空用户所有记忆: user_id={user_id}, "
            f"long_count={long_deleted_count}, short_count={short_deleted_count}"
        )
        return {
            "long": long_deleted_count,
            "short": short_deleted_count,
        }

    def _search(
            self,
            namespace: tuple[str, str],
            query: str,
            limit: int,
    ):
        """
        根据命名空间和用户请求，返回与之相关的记忆内容
        :param namespace:
        :param query:  用户请求，如果为空字符串，则返回最近更新的 limit  条记录
        :param limit:
        :return:
        """
        try:
            if query.strip():
                return self.store.search(namespace, query=query, limit=limit)
            return self.store.search(namespace, limit=limit)  # 例如查看所有记忆的时候
        except Exception:
            return self.store.search(namespace, limit=limit)

    def _find_existing(
            self,
            namespace: tuple[str, str],
            content: str,
            memory_id: str | None,
    ) -> MemoryRecord | None:
        if memory_id:  # 如果根据 memory_id 能找到记忆内容则返回，没找到则返回None（传入的 memory_id 可能有误）
            item = self.store.get(namespace, memory_id)
            return self._item_to_record(item) if item else None

        # 列出这个用户 namespace 下最近更新的 100 条记忆”，然后代码再逐条精确比对
        # 不过这种的命中率很低，价值不大
        for item in self.store.search(namespace, limit=100):
            record = self._item_to_record(item)
            if record and record.content.strip() == content:
                return record
        return None

    @staticmethod
    def _item_to_record(item) -> MemoryRecord | None:
        value = item.value if item and isinstance(item.value, dict) else {}
        content = value.get("content")
        if not content:
            return None
        memory_type = PostgresMemoryStore._normalize_memory_type(
            value.get("memory_type", "semantic")
        )
        created_at = value.get("created_at") or PostgresMemoryStore._iso_value(
            getattr(item, "created_at", None)
        )
        updated_at = value.get("updated_at") or PostgresMemoryStore._iso_value(
            getattr(item, "updated_at", None)
        )
        return MemoryRecord(
            memory_id=value.get("memory_id", item.key),
            content=content,
            context=value.get("context", ""),
            created_at=created_at,
            updated_at=updated_at,
            memory_type=memory_type,
        )

    @staticmethod
    def _normalize_memory_type(memory_type: str) -> MemoryType:
        if memory_type in MEMORY_TYPE_LABELS:
            return memory_type
        return "semantic"

    @staticmethod
    def _iso_value(value) -> str:
        if hasattr(value, "isoformat"):
            return value.isoformat(timespec="seconds")
        return str(value or "")


__all__ = [
    "PostgresMemoryStore",
    "MemoryRecord",
    "MemorySearchResult",
    "MemoryType",
    "MEMORY_TYPE_LABELS",
]
