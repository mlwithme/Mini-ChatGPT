from src.cli import run_cli
import argparse
from src.logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""
    # 初始化参数解析器，并给出一行简短说明。
    parser = argparse.ArgumentParser(description="MiniChatGPT")
    # 用户可以指定当前会话对应的 user_id。
    parser.add_argument("--user-id", default="demo-user", help="用于保存用户记忆文件的用户ID")
    # 用户可以覆盖默认模型名。
    # parser.add_argument("--model", default="qwen-plus", help="千问大模型名称")
    parser.add_argument("--model", default="qwen3.7-plus", help="千问大模型名称")
    # 用户可以指定本地记忆文件目录。
    parser.add_argument(
        "--memory-dir",
        default=".memory",
        help="保存长期记忆的文件目录",
    )
    # `--once` 用于一问一答后立即退出，适合快速演示。
    parser.add_argument(
        "--once",
        help="执行单轮问答模式",
    )
    # 日志等级保持简单，调试时用 DEBUG，日常演示用 INFO。
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志输出等级",
    )
    # 日志默认保存到本地文件，也允许用户指定其他路径。
    parser.add_argument(
        "--log-file",
        default="logs/memory-agent.log",
        help="日志保存目录",
    )
    # 返回配置好的解析器。
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level, args.log_file)
    # 从命令行解析参数。
    run_cli(args)


# cd mini-chagpt
# 通过 python main.py 直接执行。
if __name__ == "__main__":
    main()


# treeView-beta
# "mini-chatgpt"
#     "src"
#         "agetn.py"
#         "cli.py"
#         "context.py"
#         "qwen.py"
#         "store.py"
#         "tools.py"
#     "main.py"
