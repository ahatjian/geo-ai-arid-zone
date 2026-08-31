"""
分析结果一键入库 UI 组件
========================
为各分析页面提供「保存到数据下载中心」按钮, 打通
分析 → 结果库 → 统一导出 的完整闭环。

用法:
    from utils.save_ui import render_save_button
    render_save_button(
        default_name="塔里木_盐渍化_2025",
        data=result.category,      # numpy 数组 → npy
        kind="npy",
        meta={"研究区": area, "模块": "土壤盐渍化"},
        key_suffix="salinity",
    )

支持 kind: npy / csv / json / txt / md / png / jpg / tif / zip / pdf
"""

import os
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def render_save_button(
    default_name: str,
    data: Any,
    kind: str = "npy",
    meta: Optional[Dict[str, Any]] = None,
    key_suffix: str = "",
    label: str = "💾 保存到数据下载中心",
) -> bool:
    """
    渲染一个保存按钮, 点击后调用 results_store.save_result 入库。

    参数:
        default_name: 默认结果名称 (可编辑)
        data: 结果数据 (numpy 数组 / DataFrame / str / bytes)
        kind: 数据类型, 见 KIND_EXT
        meta: 附加元数据 (研究区/日期/模块等)
        key_suffix: 唯一标识, 避免页面多按钮 key 冲突
        label: 按钮文案

    返回:
        bool: 本次是否点击并保存成功
    """
    import streamlit as st
    from utils.results_store import save_result

    # 名称可编辑 (带默认值), key 需唯一
    edit_key = f"save_name_{key_suffix}"
    default_value = st.session_state.get(edit_key, default_name)

    col1, col2 = st.columns([3, 1])
    with col1:
        name = st.text_input(
            "结果名称",
            value=default_value,
            key=edit_key,
            label_visibility="collapsed",
            placeholder="输入结果名称",
        )
    with col2:
        clicked = st.button(label, key=f"save_btn_{key_suffix}")

    if clicked:
        try:
            res = save_result(name or default_name, data, kind=kind, meta=meta or {})
            st.success(f"✅ 已保存: **{res['name']}** ({res['size_human']})")
            st.caption("📦 可在「数据下载中心」页查看、下载或打包")
            return True
        except Exception as e:
            st.error(f"保存失败: {e}")
    return False


def render_save_csv_button(
    default_name: str,
    stats: Any,
    meta: Optional[Dict[str, Any]] = None,
    key_suffix: str = "",
) -> bool:
    """保存统计结果 (list[dict] 或 DataFrame) 为 CSV。"""
    import pandas as pd

    if isinstance(stats, list) and stats and isinstance(stats[0], dict):
        df = pd.DataFrame(stats)
    elif isinstance(stats, pd.DataFrame):
        df = stats
    else:
        df = pd.DataFrame(stats)
    return render_save_button(
        default_name, df, kind="csv", meta=meta, key_suffix=f"csv_{key_suffix}"
    )
