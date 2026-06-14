"""本地 JSON 版记忆存储实现。"""

# 延迟解析类型注解。
from __future__ import annotations

# 用 JSON 把记忆持久化到本地文件中。
import json
# 标准日志模块，用于记录记忆读写和检索流程。
import logging
# `math` 用于计算余弦相似度时的平方根。
import math
# `re` 同时用于安全文件名处理和简单分词。
import re
# UUID 用于生成稳定且唯一的记忆 ID。
import uuid
# dataclass 让数据结构定义更简洁。
from dataclasses import asdict, dataclass
# 时间戳记录记忆的创建和更新时间。
from datetime import datetime
# `Path` 让文件系统操作更清晰。
from pathlib import Path
# `Any` 用于描述可选的 embedding 模型对象。
from typing import Any, Literal

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
    """
    表示一条已经落盘保存的记忆。
    e.g.
  {
    "memory_id": "5f45982c-751a-411b-b4e6-a1da03ed75ad",
    "content": "用户籍贯为四川，当前常居地为上海",
    "context": "用户的地理背景信息，可能影响后续对地域相关话题（如饮食、方言、政策等）的交流",
    "created_at": "2026-06-08T19:24:04",
    "updated_at": "2026-06-08T19:24:04",
    "memory_type": "semantic",
    "embedding": [
      -0.09876544028520584,
      0.019420059397816658,...
    }

    """

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
    # 可选的向量表示，用于语义检索。
    embedding: list[float] | None = None

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


class JsonMemoryStore:
    """
    把一个用户的所有记忆保存到本地 JSON 文件中，形如
    e.g.
    [{
    "memory_id": "65864dd6-cc5f-4804-8b4d-11f3b3b35548",
    "content": "用户住在上海",
    "context": "地理位置信息，可能影响后续示例的本地化（如时区、日期格式、生活场景举例等）",
    "created_at": "2026-05-31T16:18:04",
    "updated_at": "2026-05-31T16:18:04",
    "memory_type": "semantic",
    "embedding": [
      -0.0667203739285469,
      0.025669077411293983,...]
      },
      {
    "memory_id": "13903326-25f3-4724-b21e-35b8a143352e",
    "content": "有一个好朋友叫张三",
    "context": "人际关系信息，可能用于后续个性化举例、故事化解释或社交场景模拟（如协作开发、技术分享、职业建议等）",
    "created_at": "2026-05-31T16:26:19",
    "updated_at": "2026-05-31T16:26:19",
    "memory_type": "semantic",
    "embedding": [
      -0.05820561200380325,
      0.003316354937851429,...]
      }]
    """

    def __init__(self, base_dir: str | Path, embeddings_model: Any | None = None):
        """初始化本地记忆存储。"""
        # 统一转成 `Path` 对象，便于后续处理。
        self.base_dir = Path(base_dir)
        # 如果目录不存在，就自动创建。
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # 可选保存一个向量模型，用于语义检索。
        self.embeddings_model = embeddings_model
        logger.debug(f"初始化 JSON 记忆存储: base_dir={self.base_dir}, embeddings={bool(embeddings_model)}")

    def list_memories(self, user_id: str) -> list[MemoryRecord]:
        """根据 user_id 读取某个用户的全部记忆。"""

        # 先定位当前用户对应的记忆文件。
        path = self._memory_file(user_id)
        # 文件不存在说明这个用户还没有任何记忆。
        if not path.exists():
            logger.debug(f"记忆文件不存在（第1次调用或已被清除）: user_id={user_id}, path={path}")
            return []
        # 从 JSON 文件中读出原始数据。
        raw = json.loads(path.read_text(encoding="utf-8"))
        # 把字典列表转换成 `MemoryRecord` 列表，每个元素就是一条记忆
        records = []
        for item in raw:
            item.setdefault("memory_type", "semantic")
            item["memory_type"] = self._normalize_memory_type(item["memory_type"])
            records.append(MemoryRecord(**item))
        logger.debug(f"读取用户记忆: user_id={user_id}, count={len(records)}")
        return records

    def search(self, user_id: str, query: str, limit: int = 5) -> list[MemorySearchResult]:
        """
        根据用户输入的请求，搜索与当前请求最相关的记忆。
        这里采用的是两个向量之间的余弦相似度衡量
        如果没有向量化模型或者向量化失败，使用简单词重叠作为相似度。
        """

        # 先拿到这个用户的全部记忆。
        records = self.list_memories(user_id)
        # 如果没有记忆，就直接返回空结果。
        if not records:
            logger.debug(f"没有可检索的记忆: user_id={user_id}")
            return []

        # 尝试把当前查询转成向量。
        query_embedding = self._embed(query)
        logger.debug(f"查询向量化结果: user_id={user_id}, embedded={query_embedding is not None}")
        # 对每条记忆计算与当前查询的相关度。
        scored = [
            MemorySearchResult(
                record=record,
                score=self._score(record, query, query_embedding),
            )
            for record in records
        ]
        # 按分数从高到低排序。
        scored.sort(key=lambda item: item.score, reverse=True)
        # 只保留前 `limit` 条结果。
        results = scored[:limit]  # 返回前 limit 条得分最相似的文本
        logger.debug(f"完成记忆检索: user_id={user_id}, total={len(records)}, returned={len(results)}")
        return results

    def upsert(
            self,
            user_id: str,
            content: str,
            context: str,
            *,
            memory_type: MemoryType = "semantic",
            memory_id: str | None = None,
    ) -> MemoryRecord:
        """
        新增一条记忆
        或更新一条已有的记忆。
        """
        # 先把当前用户已有的记忆全部加载出来。
        records = self.list_memories(user_id)
        # 记录当前时间，后面会用于 created_at / updated_at。
        now = datetime.now().isoformat(timespec="seconds")
        # 如果传入 memory_id 就表示将要对已有的记忆更新，否则生成新 ID 表示这是插入一条新的记忆。
        record_id = memory_id or str(uuid.uuid4())  #
        normalized_memory_type = self._normalize_memory_type(memory_type)
        # 预先计算记忆内容对应的向量。
        logger.debug(f"开始对记忆进行 embedding处理")
        embedding = self._embed(f"{content}\n{context}")

        # 先假设这次还没有命中旧记录。
        updated_record = None
        # 遍历现有记录，寻找同 ID 的旧记忆。
        for index, record in enumerate(records):
            # 不是目标记录就跳过。
            if record.memory_id != record_id:  # 遍历已有的每一条记忆，判断其 memory_id 是否等于传入的 record_id
                # 也就是说，如果你想对现有记忆更新，那就要传入一个现有记忆对应的 memory_id
                continue
            # 保留原始创建时间，只更新内容、上下文、更新时间和向量。
            updated_record = MemoryRecord(
                memory_id=record.memory_id,
                content=content,
                context=context,
                created_at=record.created_at,
                updated_at=now,
                memory_type=normalized_memory_type,
                embedding=embedding,
            )
            # 用新对象替换旧对象。
            records[index] = updated_record
            # 已经找到目标，结束循环。
            break  # 找到这个记录就结束

        # 如果没有找到旧记录，说明 record_id 说本次新生成的，表示当前是一次新增操作。
        if updated_record is None:
            updated_record = MemoryRecord(
                memory_id=record_id,
                content=content,
                context=context,
                created_at=now,
                updated_at=now,
                memory_type=normalized_memory_type,
                embedding=embedding,
            )
            # 把新记录追加到列表末尾。
            records.append(updated_record)
            logger.debug(f"新增记忆: user_id={user_id}, memory_id={record_id}, memory_type={normalized_memory_type}")
        else:
            logger.debug(f"更新记忆: user_id={user_id}, memory_id={record_id}, memory_type={normalized_memory_type}")

        # 把更新后的完整列表重新写（覆盖）回文件。
        self._save(user_id, records)
        # 返回最终写入的记录对象。
        return updated_record

    def clear(self, user_id: str) -> None:
        """清空某个用户的全部记忆。"""
        # 获取当前用户的记忆文件路径。
        path = self._memory_file(user_id)
        # 如果文件存在，就删除它。
        if path.exists():
            path.unlink()
            logger.debug(f"已清空记忆文件: user_id={user_id}, path={path}")
        else:
            logger.debug(f"记忆文件不存在，无需清空: user_id={user_id}, path={path}")

    def _memory_file(self, user_id: str) -> Path:
        """根据 user_id 拼接记忆文件路径。"""
        # 把不适合作为文件名的字符替换成下划线。
        safe_user_id = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)
        # 每个用户各自对应一个 JSON 文件。
        return self.base_dir / f"{safe_user_id}.json"

    def _save(self, user_id: str, records: list[MemoryRecord]) -> None:
        """把某个用户的记忆列表写回磁盘。"""
        path = self._memory_file(user_id)  # 再次计算这个用户的目标文件路径。
        # 把 dataclass 列表转换成 JSON 并写入磁盘。
        path.write_text(
            json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.debug(f"保存记忆文件: user_id={user_id}, path={path}, count={len(records)}")

    @staticmethod
    def _normalize_memory_type(memory_type: str) -> MemoryType:
        """把未知类型兜底为语义记忆，兼容旧数据和模型偶发输出。"""
        if memory_type in MEMORY_TYPE_LABELS:
            return memory_type  # type: ignore[return-value]
        logger.warning(f"未知记忆类型，回退为 semantic: memory_type={memory_type}")
        return "semantic"

    def _embed(self, text: str) -> list[float] | None:
        """把文本转成向量；失败时返回 `None`。"""
        # 如果没有提供向量模型，就直接放弃向量化。
        if not self.embeddings_model:
            return None
        try:
            # 调用向量模型把文本转成 embedding。
            return list(self.embeddings_model.embed_query(text))
        except Exception:
            # 如果 embedding 接口失败，则退化为关键词匹配模式。
            logger.exception("向量化失败，回退到关键词检索")
            return None

    def _score(
            self,
            record: MemoryRecord,
            query: str,
            query_embedding: list[float] | None,
    ) -> float:
        """计算一条记忆与当前查询的相关度。"""
        # 如果查询和记忆都带向量，就优先使用余弦相似度。
        if query_embedding is not None and record.embedding is not None:
            return self._cosine_similarity(query_embedding, record.embedding)
        # 否则退化为简单的关键词重叠评分。
        return self._keyword_overlap(query, record.text)

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        """计算两个向量的余弦相似度。"""
        # 如果向量为空，或者维度不一致，就无法计算相似度。
        if not left or not right or len(left) != len(right):
            return 0.0
        # 计算点积。
        dot = sum(a * b for a, b in zip(left, right))
        # 计算左侧向量长度。
        left_norm = math.sqrt(sum(a * a for a in left))
        # 计算右侧向量长度。
        right_norm = math.sqrt(sum(b * b for b in right))
        # 任意一侧长度为 0 时，相似度定义为 0。
        if left_norm == 0 or right_norm == 0:
            return 0.0
        # 返回标准余弦相似度结果。
        return dot / (left_norm * right_norm)

    @staticmethod
    def _keyword_overlap(query: str, text: str) -> float:
        """在没有向量时，使用简单词重叠作为相似度。"""
        # 把查询切成小写词集合。
        query_words = set(JsonMemoryStore._tokenize(query))
        # 把候选文本也切成小写词集合。
        text_words = set(JsonMemoryStore._tokenize(text))
        # 任意一边为空时，不存在可比较的重叠词。
        if not query_words or not text_words:
            return 0.0
        # 取两边共同出现的词。
        overlap = query_words & text_words
        # 用“查询词覆盖率”作为一个简单分数。
        return len(overlap) / len(query_words)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """把文本粗略切分成词列表，用于没有词向量，根据字符比较相似度时的情况"""
        # 用非单词字符切分文本，并过滤掉空字符串。
        return [part for part in re.split(r"\W+", text.lower()) if part]


# 指定模块对外公开的主要类型。
__all__ = [
    "JsonMemoryStore",
    "MemoryRecord",
    "MemorySearchResult",
    "MemoryType",
    "MEMORY_TYPE_LABELS",
]
