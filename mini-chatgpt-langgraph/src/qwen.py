"""封装通过 DashScope 调用 Qwen 模型的辅助函数。"""

from __future__ import annotations
import logging
import os
from dotenv import load_dotenv
# DashScope 的向量模型封装，用于记忆检索。
from langchain_community.embeddings import DashScopeEmbeddings
# DashScope 提供了 OpenAI 兼容接口，因此可以复用 ChatOpenAI。
from langchain_openai import ChatOpenAI

# DashScope OpenAI 兼容接口的默认地址。
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 默认聊天模型名称。
DEFAULT_CHAT_MODEL = "qwen-plus"
# 默认向量模型名称。
DEFAULT_EMBEDDING_MODEL = "text-embedding-v3"

logger = logging.getLogger(__name__)


def _ensure_api_key() -> str:
    """确保 DashScope API Key 已经存在。"""
    # 先尝试从 `.env` 加载环境变量。
    load_dotenv()
    # 读取 DashScope 的 API Key。
    api_key = os.getenv("DASHSCOPE_API_KEY")
    # 如果没有配置，就尽早报错并给出明确提示。
    if not api_key:
        logger.error("缺少 DASHSCOPE_API_KEY")
        raise RuntimeError(
            "Missing DASHSCOPE_API_KEY. Put it in your environment or .env before "
            "running the local demo."
        )
    # 返回已经校验过的 API Key。
    return api_key


def get_embeddings_model(model: str = DEFAULT_EMBEDDING_MODEL) -> DashScopeEmbeddings:
    """创建用于本地记忆检索的向量模型。"""
    # 先确保 API Key 已经配置。
    _ensure_api_key()
    logger.debug(f"创建 DashScope 向量模型: model={model}")
    # 构造并返回 DashScope 向量模型实例。
    return DashScopeEmbeddings(model=model)


def get_llm_model(
        model: str = DEFAULT_CHAT_MODEL,
        *,
        temperature: float = 0.3,
        base_url: str = DEFAULT_BASE_URL,
) -> ChatOpenAI:
    """创建本地记忆助手使用的聊天模型。"""
    # 先读取并校验 API Key。
    api_key = _ensure_api_key()
    logger.debug(f"创建 DashScope 聊天模型: model={model}, base_url={base_url}")
    # 返回一个指向 DashScope 的 LangChain 聊天模型对象。
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
    )


__all__ = [
    "DEFAULT_CHAT_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "get_embeddings_model",
    "get_llm_model",
]
