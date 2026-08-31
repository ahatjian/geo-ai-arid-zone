"""
系统状态与缓存管理页面 — 第 21 模块
====================================
管理平台运行时缓存 (内存 + 磁盘), 查看系统健康状态。

功能:
  - 缓存状态总览 (Streamlit 缓存 + 磁盘缓存目录)
  - 一键清空 Streamlit 内存缓存
  - 清理磁盘缓存目录 (.cache / downloads 临时文件)
  - 系统信息 (Python/依赖版本/目录占用)
"""

import os
import sys
import glob
import shutil
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from config import BASE_DIR, CACHE_DIR, DATA_DIR

st.set_page_config(page_title="系统状态", page_icon="🛠️", layout="wide")

st.title("🛠️ 系统状态与缓存管理")
st.markdown("查看平台运行状态，管理缓存释放磁盘空间")

# ============================================================
# 辅助函数
# ============================================================

def _dir_size(path: str) -> tuple:
    """返回 (大小字节, 文件数)"""
    if not os.path.exists(path):
        return 0, 0
    total = 0
    count = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
                count += 1
            except OSError:
                pass
    return total, count


def _human_size(num: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num < 1024 or unit == "GB":
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} GB"


def _clear_dir_contents(path: str) -> int:
    """清空目录内容, 返回删除文件数"""
    if not os.path.exists(path):
        return 0
    removed = 0
    for entry in os.listdir(path):
        p = os.path.join(path, entry)
        try:
            if os.path.isfile(p) or os.path.islink(p):
                os.remove(p)
                removed += 1
            elif os.path.isdir(p):
                shutil.rmtree(p)
                removed += 1
        except OSError:
            pass
    return removed


# ============================================================
# Tab 1: 缓存状态
# ============================================================
tab_cache, tab_system = st.tabs(["💾 缓存管理", "ℹ️ 系统信息"])

with tab_cache:
    st.subheader("📊 缓存状态总览")

    # 目录列表
    dirs_to_show = [
        ("缓存目录 (.cache)", os.path.join(BASE_DIR, ".cache"), "STAC 搜索/预览缓存"),
        ("下载目录 (downloads)", os.path.join(BASE_DIR, "downloads"), "历史下载文件"),
        ("数据目录 (data)", os.path.join(BASE_DIR, "data"), "平台数据文件"),
        ("结果库 (results)", os.path.join(BASE_DIR, "results"), "用户保存的分析结果"),
    ]

    col_headers = st.columns([2, 1, 1, 2])
    col_headers[0].markdown("**目录**")
    col_headers[1].markdown("**大小**")
    col_headers[2].markdown("**文件数**")
    col_headers[3].markdown("**用途**")

    dir_sizes = {}
    for name, path, purpose in dirs_to_show:
        size, count = _dir_size(path)
        dir_sizes[path] = (size, count)
        c1, c2, c3, c4 = st.columns([2, 1, 1, 2])
        c1.markdown(f"`{name}`")
        c2.markdown(f"**{_human_size(size)}**")
        c3.markdown(str(count))
        c4.caption(purpose)

    st.divider()

    # Streamlit 内存缓存
    st.subheader("🧠 Streamlit 内存缓存")
    st.caption("分析结果、STAC 搜索等会缓存在内存中加速重复访问")
    if st.button("🗑️ 清空内存缓存", key="clear_mem", type="primary"):
        st.cache_data.clear()
        st.success("✅ 内存缓存已清空")
        st.rerun()

    st.divider()

    # 磁盘缓存清理
    st.subheader("💿 磁盘缓存清理")
    st.caption("清理下载目录中的临时文件释放空间（.cache 目录为 STAC 预览缓存，清理后首次访问会重新下载）")

    clean_targets = []
    for name, path, _ in dirs_to_show:
        if name != "结果库 (results)":  # 结果库不提供清理
            size, _ = dir_sizes.get(path, (0, 0))
            if size > 0:
                clean_targets.append((name, path))

    if not clean_targets:
        st.success("✅ 所有可清理目录均为空，无需清理")
    else:
        for name, path in clean_targets:
            size, _ = dir_sizes.get(path, (0, 0))
            if st.button(f"🗑️ 清理 {name} ({_human_size(size)})", key=f"clean_{name}"):
                removed = _clear_dir_contents(path)
                st.success(f"✅ 已清理 {name}，删除 {removed} 个文件")
                st.rerun()

    st.divider()
    st.info("💡 **提示**: 清理缓存不会影响已保存到「数据下载中心」的结果；磁盘缓存只是加速数据加载的临时文件。")

# ============================================================
# Tab 2: 系统信息
# ============================================================
with tab_system:
    st.subheader("🖥️ 运行环境")

    import platform
    sys_info = [
        ("操作系统", f"{platform.system()} {platform.release()}"),
        ("Python", platform.python_version()),
        ("运行目录", BASE_DIR),
        ("进程 PID", str(os.getpid())),
    ]
    for k, v in sys_info:
        c1, c2 = st.columns([1, 3])
        c1.markdown(f"**{k}**")
        c2.markdown(f"`{v}`")

    st.divider()
    st.subheader("📦 关键依赖")

    deps = [
        "streamlit", "numpy", "pandas", "rasterio", "geopandas",
        "pystac_client", "planetary_computer", "torch", "onnxruntime",
        "statsmodels", "sklearn", "plotly", "matplotlib", "reportlab",
    ]
    dep_cols = st.columns(3)
    for i, dep in enumerate(deps):
        try:
            mod = __import__(dep)
            ver = getattr(mod, "__version__", "?")
            status = f"✅ {dep} {ver}"
        except ImportError:
            status = f"❌ {dep}"
        with dep_cols[i % 3]:
            st.markdown(status)

    st.divider()
    st.subheader("🛰️ 数据源连通性")
    if st.button("🔍 测试 STAC 连接", key="test_stac"):
        try:
            from pystac_client import Client
            cat = Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1", timeout=15
            )
            st.success(f"✅ STAC 连接正常 ({cat.title})")
        except Exception as e:
            st.error(f"❌ STAC 连接失败: {type(e).__name__}: {str(e)[:120]}")

    if st.button("🧠 测试 DeepSeek AI", key="test_ai"):
        from utils.llm import is_llm_available
        if is_llm_available():
            try:
                r = __import__("utils.llm", fromlist=["query_deepseek"]).query_deepseek(
                    "测试连接, 回复'正常'即可"
                )
                st.success(f"✅ DeepSeek 连接正常 (解析方式: {r.get('method')})")
            except Exception as e:
                st.error(f"❌ DeepSeek 调用失败: {str(e)[:120]}")
        else:
            st.warning("⚠️ 未配置 DEEPSEEK_API_KEY，AI 功能使用模板匹配模式")
