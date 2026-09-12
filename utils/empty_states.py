"""
空状态引导组件 — 无数据时给出明确下一步
==========================================
统一各页面的"无数据"提示: 不只说"没有"，而是给出可操作的排查清单,
降低用户(尤其演示场景)的困惑。

用法:
    from utils.empty_states import no_image_guidance
    if not results:
        no_image_guidance(st)
        st.stop()
"""


def no_image_guidance(st, context: str = "分析") -> None:
    """
    标准化的"未找到影像"空状态引导。

    参数:
        st: streamlit 模块
        context: 上下文描述 (如 "水体监测")
    """
    st.warning(f"⚠️ 未找到符合条件的影像 — {context}无法继续")
    st.markdown("""
    **🔍 建议按顺序排查：**

    1. **放宽云量阈值** — 干旱区夏季云少，可设 20-30%；冬季可放宽到 50%
    2. **扩大时间范围** — 至少覆盖一个完整季节（如 6-9 月植被生长期）
    3. **更换数据源** — Sentinel-2 重访 5 天（推荐）；Landsat 重访 16 天但 1982 年至今
    4. **检查研究区** — 范围过小可能无覆盖影像，可扩大到盆地级
    5. **核实日期** — 结束日期需晚于开始日期，且不晚于今天
    """)


def no_data_guidance(st, context: str = "分析", source_page: str = "") -> None:
    """
    "尚无数据"空状态引导 (引导用户先去其他页面完成分析)。

    参数:
        st: streamlit 模块
        context: 当前功能名
        source_page: 数据来源页面提示 (如 "植被分析页")
    """
    hint = f"，或先在「{source_page}」完成分析" if source_page else ""
    st.info(f"📭 {context}尚无数据 — 请先上传 GeoTIFF 数据{hint}")
    st.caption("💡 数据可从「数据浏览」页搜索下载，或直接上传本地 GeoTIFF 文件")


def no_result_guidance(st, context: str = "矢量化", suggestions: list = None) -> None:
    """"结果为空"空状态引导 (分析执行了但无有效结果)。"""
    st.warning(f"⚠️ {context}结果为空")
    if suggestions:
        st.markdown("**建议：**")
        for i, s in enumerate(suggestions, 1):
            st.markdown(f"{i}. {s}")
    else:
        st.caption("💡 请检查输入数据有效性，或调整参数后重试")
