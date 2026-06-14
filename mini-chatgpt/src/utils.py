"""放置模型加载相关的辅助函数。"""

# 标准日志模块，用于记录模型加载路径。
import logging

# LangChain 的通用模型加载器。
from langchain.chat_models import init_chat_model
# 这里返回的是兼容 LangChain 聊天接口的模型对象。
from langchain_core.language_models import BaseChatModel

# 项目内自定义的 Qwen 加载逻辑。
from .qwen import get_llm_model

logger = logging.getLogger(__name__)


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """统一加载聊天模型。

    如果传入的是 `qwen-plus` 这种不带提供方前缀的名称，
    默认按 Qwen 模型处理。

    如果传入的是 `dashscope/qwen-plus` 或 `qwen/qwen-plus`，
    也走项目内的 Qwen 封装。

    其他形如 `provider/model` 的写法，则交给 LangChain 的通用加载器。
    """
    # 如果没有 `/`，就按“裸模型名”处理，默认认为它是 Qwen 模型。
    if "/" not in fully_specified_name:
        logger.debug(f"加载 Qwen 聊天模型: model={fully_specified_name}")
        return get_llm_model(model=fully_specified_name)

    # 把 `provider/model` 这种格式拆成两部分。
    provider, model = fully_specified_name.split("/", maxsplit=1)
    # DashScope 和 Qwen 这两类前缀都走本项目的自定义封装。
    if provider in {"dashscope", "qwen"}:
        logger.debug(f"加载 Qwen 聊天模型: provider={provider}, model={model}")
        return get_llm_model(model=model)
    # 其余情况回退到 LangChain 的通用加载方式。
    logger.debug(f"通过 LangChain 加载聊天模型: provider={provider}, model={model}")
    return init_chat_model(model, model_provider=provider)
