"""
结构化日志模块
==============
平台可观测性基础 — 统一日志 + API 调用追踪

所有记录写入名为 "geo-ai" 的统一 logger (带时间戳的 StreamHandler):
Streamlit 中会出现在运行终端, 部署到服务器时可被日志采集器接管。

使用:
  from utils.logging_config import log_info, log_error, log_api_call, LogTimer

  with LogTimer("STAC search"):
      results = search_images(...)
  # 自动记录: [TIMER] STAC search: 2.34s

  log_api_call("DeepSeek", 1.23, success=True, status="第 1 次尝试")
  log_error("工作流", exc, context={"step": "download"})

注: 早期版本文档写的是 `from utils.logging_config import log`, 但本模块
从未导出过名为 `log` 的对象 —— 此处已更正为实际的具名函数。
"""

import time
import logging
from functools import wraps
from typing import Callable, Optional, Dict, Any

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
