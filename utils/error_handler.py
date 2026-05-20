"""
全局错误处理与重试机制
提供 Streamlit 装饰器、重试逻辑、进度包装器

用于: 所有 pages/*.py 和 utils/*.py 的稳定性增强
"""

import time
import functools
import traceback
from typing import Optional, Callable, Any, TypeVar, Tuple, Dict
import warnings

warnings.filterwarnings("ignore")

# 泛型类型
F = TypeVar("F", bound=Callable[..., Any])

# ============================================
# 重试装饰器
# ============================================

def with_retry(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple = (Exception,),
    on_retry: Optional[Callable] = None,
    silent: bool = False,
):
    """
    指数退避重试装饰器

    用法:
        @with_retry(max_retries=3, exceptions=(ConnectionError, TimeoutError))
        def fetch_data(url):
            ...

    参数:
        max_retries: 最大重试次数
        backoff_factor: 退避因子 (delay = initial_delay * backoff_factor^attempt)
        initial_delay: 初始延迟 (秒)
        max_delay: 最大延迟 (秒)
        exceptions: 要重试的异常类型元组
        on_retry: 重试回调 callback(attempt, exception, delay)
        silent: 是否抑制重试日志
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt >= max_retries:
                        break  # 用完重试次数

                    delay = min(initial_delay * (backoff_factor ** attempt), max_delay)

                    if not silent:
                        warnings.warn(
                            f"[重试 {attempt + 1}/{max_retries}] {func.__name__} 失败: {e}. "
                            f"{delay:.1f}s 后重试..."
                        )

                    if on_retry:
                        on_retry(attempt + 1, e, delay)

                    time.sleep(delay)

            # 所有重试都失败
            raise RuntimeError(
                f"{func.__name__} 在 {max_retries} 次重试后仍然失败: {last_exception}"
            ) from last_exception

        return wrapper  # type: ignore

    return decorator


# ============================================
# 安全的函数执行包装器 (非装饰器版本)
# ============================================

def safe_execute(
    func: Callable,
    *args,
    fallback_value: Any = None,
    error_message: str = "执行失败",
    on_error: Optional[Callable] = None,
    **kwargs,
) -> Tuple[Any, Optional[str]]:
    """
    安全执行函数，返回 (结果, 错误信息)

    用法:
        result, error = safe_execute(download_image, url, fallback_value=None)

    返回:
        (result, error_str_or_None)
    """
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        error_str = f"{error_message}: {str(e)}"
        if on_error:
            on_error(e, error_str)
        return fallback_value, error_str


# ============================================
# Streamlit 页面错误边界
# ============================================

class StreamlitErrorBoundary:
    """
    Streamlit 页面级错误边界

    用法:
        with StreamlitErrorBoundary("水体监测", st=st):
            # 页面主体代码
            ...
    """

    def __init__(self, page_name: str, st=None, show_traceback: bool = False):
        """
        参数:
            page_name: 页面名称
            st: streamlit 模块 (在页面内传入)
            show_traceback: 是否显示完整 traceback (仅开发模式)
        """
        self.page_name = page_name
        self.st = st
        self.show_traceback = show_traceback
        self._error_occurred = False

    def __enter__(self):
        self._error_occurred = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._error_occurred = True

            if self.st is not None:
                self.st.error(f"## ⚠️ {self.page_name} 发生错误")
                self.st.error(f"**{exc_type.__name__}**: {str(exc_val)}")

                if self.show_traceback:
                    with self.st.expander("🔍 错误详情"):
                        self.st.code(traceback.format_exc())

                # 恢复建议
                self.st.info(
                    "💡 **建议**: 请检查输入数据是否完整, "
                    "或刷新页面重试。如问题持续存在, 请联系管理员。"
                )
            else:
                import sys
                print(f"[{self.page_name}] 错误: {exc_val}", file=sys.stderr)
                traceback.print_exception(exc_type, exc_val, exc_tb)

            return True  # 抑制异常

        return False

    @property
    def error_occurred(self) -> bool:
        return self._error_occurred


# ============================================
# 装饰器: 给 Streamlit 页面函数加错误保护
# ============================================

def streamlit_safe(page_name: str, show_traceback: bool = False):
    """
    装饰器: 为 Streamlit 页面渲染函数提供错误保护

    用法:
        @streamlit_safe("水体监测")
        def render_water_monitoring():
            # 需要传入 streamlit 模块
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                import streamlit as st
                st.error(f"## ⚠️ {page_name} 发生错误")
                st.error(f"**{type(e).__name__}**: {str(e)}")

                if show_traceback:
                    with st.expander("🔍 错误详情"):
                        st.code(traceback.format_exc())

                st.info("💡 请检查输入数据, 或刷新页面重试。")
                return None

        return wrapper  # type: ignore

    return decorator


# ============================================
# 进度条包装器
# ============================================

class StreamlitProgress:
    """
    Streamlit 友好进度条包装器

    用法:
        progress = StreamlitProgress(st, "正在处理...", total=100)
        for i in range(100):
            progress.update(i + 1)
        progress.close()
    """

    def __init__(
        self,
        st,
        description: str = "处理中...",
        total: int = 100,
        show_percentage: bool = True,
    ):
        self.st = st
        self.description = description
        self.total = total
        self.show_percentage = show_percentage
        self._bar = None
        self._text = None

    def __enter__(self):
        self._bar = self.st.progress(0, text=f"{self.description} 0%")
        return self

    def update(self, current: int, extra_text: str = ""):
        if self._bar is None:
            self._bar = self.st.progress(0)

        fraction = min(current / max(self.total, 1), 1.0)

        if self.show_percentage:
            pct = int(fraction * 100)
            text = f"{self.description} {pct}%"
            if extra_text:
                text += f" ({extra_text})"
            self._bar.progress(fraction, text=text)
        else:
            self._bar.progress(fraction)

    def __exit__(self, *args):
        if self._bar is not None:
            self._bar.empty()

    def close(self):
        if self._bar is not None:
            self._bar.empty()
            self._bar = None


# ============================================
# 检查必需依赖
# ============================================

DEPENDENCY_CHECKLIST = {
    "streamlit": ("1.35+", "Web UI 框架", "pip install streamlit"),
    "leafmap": ("0.36+", "地图可视化", "pip install leafmap"),
    "rasterio": ("1.3+", "GeoTIFF 读写", "pip install rasterio"),
    "numpy": ("1.24+", "数值计算", "pip install numpy"),
    "pandas": ("2.0+", "数据分析", "pip install pandas"),
    "plotly": ("5.18+", "交互图表", "pip install plotly"),
    "matplotlib": ("3.7+", "静态绑图", "pip install matplotlib"),
    "torch": ("2.0+", "深度学习", "pip install torch"),
    "pystac_client": ("0.8+", "STAC 搜索", "pip install pystac-client"),
    "planetary_computer": ("1.0+", "PC 数据源", "pip install planetary-computer"),
    "geopandas": ("0.14+", "矢量分析", "pip install geopandas"),
    "xarray": ("2024.1+", "多维数据", "pip install xarray"),
    "rioxarray": ("0.15+", "栅格 xarray", "pip install rioxarray"),
    "pymannkendall": ("1.4+", "MK 趋势检验", "pip install pymannkendall"),
    "onnxruntime": ("1.18+", "ONNX 推理加速", "pip install onnxruntime"),
    "scipy": ("1.11+", "科学计算", "pip install scipy"),
    "PIL": ("10.0+", "图像处理", "pip install Pillow"),
    "shapely": ("2.0+", "几何计算", "pip install shapely"),
}


def check_dependencies(st=None) -> Dict[str, bool]:
    """
    检查所有依赖是否安装

    参数:
        st: streamlit 模块 (可选, 用于显示结果)

    返回:
        dict: {package_name: is_installed}
    """
    import importlib

    results = {}

    for pkg_name, (version, description, install_cmd) in DEPENDENCY_CHECKLIST.items():
        try:
            mod = importlib.import_module(pkg_name)
            installed_version = getattr(mod, "__version__", "?")
            results[pkg_name] = True

            if st:
                st.success(f"✅ {pkg_name} {installed_version} — {description}")

        except ImportError:
            results[pkg_name] = False

            if st:
                st.error(f"❌ {pkg_name} 未安装 — {description}")
                st.code(install_cmd, language="bash")

    return results


# ============================================
# 内存使用监控
# ============================================

def get_memory_usage_mb() -> float:
    """获取当前进程内存使用 (MB) — Windows/Linux 通用"""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 * 1024), 2)
    except ImportError:
        return -1.0


# ============================================
# 安全的临时文件清理
# ============================================

def safe_cleanup(paths: list, st=None):
    """
    安全清理临时文件列表

    参数:
        paths: 文件路径列表
        st: streamlit 模块 (可选)
    """
    import os

    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass  # 清理失败不影响主流程
