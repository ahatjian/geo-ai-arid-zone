"""
共享缓存装饰器
==============
从 11 个 utils 模块中提取的公共 _cache 工厂函数

所有使用 @_cache(ttl) 的模块现在改为:
  from utils.cache import cache_decorator
  _cache = cache_decorator
"""

from typing import Callable

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False


def _make_cache_decorator(ttl: int) -> Callable:
    """
    创建条件缓存装饰器

    在 Streamlit 环境中使用 st.cache_data, 否则返回 no-op

    参数:
        ttl: 缓存过期时间 (秒)

    返回:
        装饰器函数
    """
    if _HAS_STREAMLIT:
        return st.cache_data(ttl=ttl, show_spinner=True)
    else:
        def noop(func: Callable) -> Callable:
            return func
        return noop
