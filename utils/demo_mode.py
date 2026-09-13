"""离线演示模式 — 全局开关

平台通常依赖 Microsoft Planetary Computer 拉取卫星影像。启用离线演示模式后,
utils/pc_data.search_images 会分派到 utils/demo_data 生成本地合成影像,
使断网 / 弱网 / 答辩现场仍能跑通"搜索 → 分析 → 导出"全流程。

开关状态存放在环境变量 GEOAI_DEMO_MODE ("1" / "0") 而非 session_state,
因为 pc_data、demo_data 等底层模块不依赖 Streamlit, 环境变量是唯一能
贯穿各层的通道。session_state 会随页面切换重跑而失效, 不适合承载它。

用法: 在 st.navigation 的入口文件 (app.py) 中渲染一次开关即可全站生效 ——
入口脚本是各页面的共享 frame, 在其中渲染的 sidebar 组件出现在所有页面。
"""

from __future__ import annotations

import os

__all__ = ["DEMO_MODE_ENV", "is_demo_mode", "set_demo_mode", "render_demo_mode_toggle"]

# 环境变量名 — pc_data / demo_data 据此分派数据源
DEMO_MODE_ENV = "GEOAI_DEMO_MODE"


def is_demo_mode() -> bool:
    """当前是否处于离线演示模式。"""
    return os.environ.get(DEMO_MODE_ENV, "0") == "1"


def set_demo_mode(enabled: bool) -> None:
    """设置离线演示模式开关 (同步写入环境变量)。"""
    os.environ[DEMO_MODE_ENV] = "1" if enabled else "0"


def render_demo_mode_toggle(key: str = "global_demo_mode") -> bool:
    """渲染离线演示模式开关 (需在侧边栏上下文中调用), 返回当前状态。

    参数:
        key: widget key, 入口文件固定用默认值即可

    返回:
        bool: 开关是否开启
    """
    import streamlit as st

    enabled = st.checkbox(
        "🧪 离线演示模式",
        value=is_demo_mode(),
        key=key,
        help="使用本地生成的模拟影像, 断网时仍可完成搜索、分析、导出全流程",
    )
    set_demo_mode(enabled)

    if enabled:
        st.caption("✅ 已启用本地演示影像, 当前不访问外部卫星服务")

    return enabled
