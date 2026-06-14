"""定义 Agent 运行时需要的上下文配置。"""
# dataclass 让配置对象更简洁清晰。
from dataclasses import dataclass, field
# 默认系统提示词单独放在 prompts 模块中。
from . import prompts


# `kw_only=True` 强制只能用关键字参数初始化
@dataclass(kw_only=True)
class Context:
    """本地记忆助手的运行配置。"""

    # 用户 ID 决定读写哪一份记忆文件。
    user_id: str = "default"
    """需要被保存和读取记忆的用户标识，每个用户一个独立的id"""

    # 模型名称会传给 DashScope 的 OpenAI 兼容接口。
    model: str = field(
        default="qwen-plus",
        metadata={
            "description": "传给 DashScope OpenAI 兼容接口的聊天模型名，例如 qwen-plus。"
        },
    )

    # 这个目录用于存放本地 JSON 记忆文件。
    memory_dir: str = field(
        default=".memory",
        metadata={
            "description": "本地演示程序用来持久化记忆文件的目录。"
        },
    )

    # 直接复用统一定义的系统提示词模板。
    system_prompt: str = prompts.SYSTEM_PROMPT  # 导入带长期记忆智能体的默认提示词
