"""项目内 logging 的简单配置。"""

from __future__ import annotations

import logging
from pathlib import Path



class ConsoleFilter(logging.Filter):
    def filter(self, record):
        return not getattr(record, "user_input", False)


def configure_logging(level: str = "INFO", log_file: str = "logs/memory-agent.log") -> None:
    """配置命令行程序的日志输出。"""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOG_FORMAT = "%(message)s"
    if level == "DEBUG":
        DEFAULT_LOG_FORMAT = "[%(asctime)s][%(filename)s:%(lineno)d]: %(message)s"
    log_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.addFilter(ConsoleFilter())
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=log_level,
        handlers=[console_handler, file_handler],
        force=True,
    )


__all__ = ["configure_logging"]
