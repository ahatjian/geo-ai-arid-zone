"""
结构化日志模块
==============
平台可观测性基础 — 统一日志 + API 调用追踪

使用:
  from utils.logging_config import log, LogTimer

  with LogTimer("STAC search"):
      results = search_images(...)
  # 自动记录: [TIMER] STAC search: 2.34s

  log.info("模块名", message="操作说明", **kwargs)
"""

import time
import logging
import warnings
from functools import wraps
from typing import Callable, Optional, Dict, Any
from contextlib import contextmanager

# 统一 logger
_logger = logging.getLogger("geo-ai")
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    ))
    _logger.addHandler(_handler)


class LogTimer:
    """上下文管理器 — 自动计时并记录"""

    def __init__(self, operation: str):
        self.operation = operation
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        _logger.info(f"[TIMER] {self.operation}: {self.elapsed:.2f}s")


def log_api_call(api_name: str, duration: float, success: bool = True,
                 status: str = "", extra: Optional[Dict] = None):
    """记录外部 API 调用"""
    icon = "✅" if success else "❌"
    _logger.info(
        f"[API] {icon} {api_name} | {duration:.2f}s"
        + (f" | {status}" if status else "")
    )


def log_error(module: str, error: Exception, context: Dict[str, Any] = None):
    """记录错误"""
    ctx = f" | {context}" if context else ""
    _logger.error(f"[ERROR] {module}: {type(error).__name__}: {error}{ctx}")


def log_warning(module: str, message: str):
    """记录警告"""
    _logger.warning(f"[WARN] {module}: {message}")


def log_info(module: str, message: str):
    """记录信息"""
    _logger.info(f"[INFO] {module}: {message}")


def with_logging(module_name: str):
    """装饰器 — 自动记录函数调用耗时"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with LogTimer(f"{module_name}.{func.__name__}"):
                return func(*args, **kwargs)
        return wrapper
    return decorator
