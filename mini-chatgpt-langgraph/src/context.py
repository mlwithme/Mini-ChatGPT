from dataclasses import dataclass, field


@dataclass(kw_only=True)
class Context:
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
    thread_id: str = field(
        default="demo_user_001",
        metadata={
            "description": "默认用户id"
        },
    )

    memory_root: str = field(
        default="demo_memory_root",
        metadata={
            "description": "PostgresStore 中长期记忆 namespace 的根节点"
        },
    )
