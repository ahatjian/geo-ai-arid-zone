"""
冰冻圈分析页面 — 积雪 / 冰川 / 冻土
====================================
基于 Sentinel-2/Landsat 多光谱数据
指标: NDSI, Snow Cover, Glacier Boundary, Snow Line, Frozen Ground
"""
import streamlit as st
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta
from io import BytesIO
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import STUDY_AREAS, COLLECTIONS, COLORMAPS, CACHE_CONFIG
from utils.error_handler import StreamlitErrorBoundary
from utils.aoi import render_aoi_selector
from utils.pc_data import (
    search_images, get_rgb_preview_cached,
    download_multiband,
)

st.set_page_config(page_title="冰冻圈分析", page_icon="❄️", layout="wide")

# ============================================================
# 样式
# ============================================================
st.markdown("""
<style>
.cryo-stat-box {
    background: #1a1a2e;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
}
.cryo-stat-box .label { font-size: 11px; color: #8ab4f8; margin-bottom: 4px; }
.cryo-stat-box .value { font-size: 20px; font-weight: 700; color: #e8eaed; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.title("❄️ 冰冻圈分析设置")

    st.subheader("研究区")
    area_name, bbox, center, source, area_info = render_aoi_selector(
        default_area="天山北坡",
        help_text="推荐天山北坡/柴达木盆地 (冰川发育区)",
        key_prefix="cryo",
    )
    if bbox is None:
        st.stop()

    st.divider()

    st.subheader("数据源")
    satellite = st.selectbox(
        "卫星数据",
        list(COLLECTIONS.keys()),
        format_func=lambda x: f"{x} ({COLLECTIONS[x]['resolution']}m)",
    )
    st.caption(COLLECTIONS[satellite]["description"])

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", date.today() - timedelta(days=180))
    with col2:
        end_date = st.date_input("结束日期", date.today())

    cloud_cover = st.slider("最大云量 (%)", 0, 100, 10,
        help="积雪分析需要低云量影像")
    max_items = st.slider("最大影像数", 1, 15, 5,
        help="冰川分析通常 1-2 景清晰影像即可")

    st.divider()

    st.subheader("分析选项")
    use_ndsi = st.checkbox("NDSI 积雪指数", value=True)
    use_snow_class = st.checkbox("积雪覆盖分类 (4级)", value=True)
    use_glacier = st.checkbox("冰川边界提取", value=True,
        help="NIR/SWIR 比值 + NDSI + NDVI 三重过滤")
    use_frozen = st.checkbox("冻土活动层分析", value=True)
    use_timeline = st.checkbox("多时相雪盖变化", value=False,
        help="需要多景影像")

    st.divider()

    # 高级设置
    with st.expander("⚙️ 高级设置"):
        ndsi_threshold = st.slider("NDSI 积雪阈值", 0.2, 0.8, 0.4, 0.05,
            help="> 此值视为积雪")
        pixel_size = st.number_input("像元大小 (m)", value=10.0, min_value=1.0)

    search_clicked = st.button(
        "🔍 搜索影像 & 分析", type="primary", width="stretch"
    )

# ============================================================
# 主页面
# ============================================================
st.title("❄️ 冰冻圈遥感分析")
st.markdown(
    f"**研究区: {area_name}** | "
    f"数据源: {satellite} | "
    f"日期: {start_date} → {end_date}"
)

if search_clicked:
    # bbox 已由研究区/AOI 选择组件提供

    # ---- Step 1: 搜索影像 ----
    with st.spinner("🔍 正在搜索影像..."):
        try:
            results = search_images(
                bbox=bbox,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                collection=satellite,
                cloud_cover_max=cloud_cover,
                max_items=max_items,
            )
        except Exception as e:
            st.error(f"搜索失败: {e}")
            results = []

    if not results:
        st.warning("⚠️ 未找到符合条件的影像。降低云量阈值试试？")
        st.stop()

    st.success(f"✅ 找到 **{len(results)}** 景影像")

    # ---- Step 2: 影像选择 ----
    st.subheader("📋 选择分析影像")
    cols = st.columns(3)
    selected_items = []
    for i, r in enumerate(results):
        with cols[i % 3]:
            try:
                preview = get_rgb_preview_cached(r["id"], collection=satellite, width=256)
            except Exception:
                preview = None

            if preview:
                st.image(preview)
            else:
                st.markdown(
                    '<div style="width:100%;height:120px;background:#1a1a2e;'
                    'border:1px dashed #444;border-radius:6px;display:flex;'
                    'align-items:center;justify-content:center;color:#666">'
                    '无预览</div>',
                    unsafe_allow_html=True,
                )

            selected = st.checkbox(
                f"**{r['datetime']}** ☁️{r['cloud_cover']}%",
                key=f"sel_cryo_{r['id']}",
                value=i < 3,
            )
            if selected:
                selected_items.append(r)

    n_selected = len(selected_items)
    if n_selected == 0:
        st.warning("⚠️ 请至少选择 1 景影像")
        st.stop()

    st.info(f"📊 已选择 **{n_selected}** 景影像")

    # ---- Step 3: 下载 & 分析 ----
    progress_bar = st.progress(0)
    status_text = st.empty()

    all_ndsi = []
    all_bands = []
    image_labels = []

    for idx, item in enumerate(selected_items):
        pct = (idx + 1) / n_selected
        progress_bar.progress(pct)
        status_text.text(f"⬇️ 分析中... [{idx+1}/{n_selected}] {item['datetime']}")

        try:
            tmp_path = os.path.join(tempfile.gettempdir(), f"cryo_{item['id'][:12]}.tif")
            tif_path = download_multiband(item["item"], tmp_path, collection=satellite)

            if tif_path and os.path.exists(tif_path):
                import rasterio
                with rasterio.open(tif_path) as src:
                    bands_data = src.read().astype(np.float32)

                # 计算 NDSI
                green = bands_data[1]
                swir1 = bands_data[4]

                # 自动缩放
                if np.nanmedian(green) > 10:
                    green = green / 10000.0
                    swir1 = swir1 / 10000.0

                denom = green + swir1
                ndsi = np.where(denom > 1e-6, (green - swir1) / denom, 0.0)
                ndsi = np.clip(ndsi, -1.0, 1.0)

                all_ndsi.append(ndsi)
                all_bands.append(bands_data)
                image_labels.append(item["datetime"])

                try:
                    os.remove(tif_path)
                except Exception:
                    pass

        except Exception as e:
            st.warning(f"⚠️ {item['datetime']} 处理失败: {e}")

    progress_bar.progress(1.0)
    status_text.text("✅ 分析完成")

    if not all_ndsi:
        st.error("❌ 没有成功处理的影像")
        st.stop()

    st.success(f"✅ 成功分析 **{len(all_ndsi)}** 景影像")

    # ====================================================
    # 结果展示
    # ====================================================

    # 取最新 (第一景) 作为主分析影像
    ndsi_main = all_ndsi[0]
    bands_main = all_bands[0]
    main_date = image_labels[0]

    # 导入冰冻圈模块
    from utils.cryosphere import (
        calc_snow_cover, extract_glacier_mask,
        estimate_snow_line, analyze_frozen_ground,
        compute_snow_cover_stats, compute_glacier_stats,
        SNOW_COVER_CLASSES, GLACIER_CLASSES, assess_cryosphere,
    )

    # 一站式评估
    with st.spinner("⏳ 冰冻圈评估中..."):
        result = assess_cryosphere(
            bands_main, satellite=satellite,
            pixel_size_m=pixel_size,
            snow_threshold=ndsi_threshold,
            extract_glacier=use_glacier,
        )

    # ---- 汇总卡片 ----
    st.subheader("📊 冰冻圈评估汇总")
    s = result.summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_snow = s["total_snow_cover_ratio"]
        st.metric("积雪覆盖率 🌨️", f"{total_snow*100:.1f}%")
    with col2:
        st.metric("主导雪盖类型", s["dominant_snow_class"])
    with col3:
        st.metric("NDSI 均值", f"{s['ndsi_mean']:.4f}")
    with col4:
        st.metric("冻土状态 🧊", s["frozen_ground"]["state"])

    st.divider()

    # ---- 分析标签页 ----
    tab_names = []
    if use_ndsi:
        tab_names.append("NDSI 积雪指数")
    if use_snow_class:
        tab_names.append("积雪分类")
    if use_glacier:
        tab_names.append("冰川边界")
    if use_frozen:
        tab_names.append("冻土分析")

    if not tab_names:
        tab_names = ["NDSI 积雪指数"]  # 默认至少一个

    tabs = st.tabs(tab_names)

    tab_idx = 0

    # Tab 1: NDSI
    if use_ndsi:
        with tabs[tab_idx]:
            tab_idx += 1
            st.caption("归一化积雪指数 — NDSI > 0.4 为积雪")

            col_m, col_s = st.columns([3, 1])
            with col_m:
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(ndsi_main, cmap="Blues_r", vmin=-0.3, vmax=0.9)
                ax.set_title(f"NDSI — {area_name} ({main_date})", fontsize=13)
                ax.axis("off")
                cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
                cbar.set_label("NDSI", fontsize=10)
                # 标注阈值线
                cbar.ax.axhline(y=ndsi_threshold, color="red", linestyle="--", linewidth=1)
                buf = BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                plt.close()
                buf.seek(0)
                st.image(Image.open(buf))

            with col_s:
                ndsi_valid = ndsi_main[np.isfinite(ndsi_main)]
                st.metric("均值", f"{np.nanmean(ndsi_valid):.4f}")
                st.metric("标准差", f"{np.nanstd(ndsi_valid):.4f}")
                snow_px = int((ndsi_main > ndsi_threshold).sum())
                total_px = int(np.isfinite(ndsi_main).sum())
                st.metric("积雪像元", f"{snow_px:,}")
                st.metric("积雪比例", f"{snow_px/max(total_px,1)*100:.1f}%")

    # Tab 2: 积雪分类
    if use_snow_class:
        with tabs[tab_idx]:
            tab_idx += 1
            st.caption("4级积雪覆盖分类: 裸地 → 薄雪 → 积雪 → 冰川/粒雪")

            col_m, col_s = st.columns([3, 2])
            with col_m:
                # 色彩映射
                color_map = {c["code"]: c["color"] for c in SNOW_COVER_CLASSES}
                cat_rgb = np.zeros((*result.snow_cover.shape, 3), dtype=np.uint8)
                from matplotlib.colors import to_rgb
                for cls in SNOW_COVER_CLASSES:
                    rgb = tuple(int(c * 255) for c in to_rgb(cls["color"]))
                    cat_rgb[result.snow_cover == cls["code"]] = rgb

                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(cat_rgb)
                ax.set_title(f"积雪覆盖分类 — {area_name}", fontsize=13)
                ax.axis("off")
                # 图例
                from matplotlib.patches import Patch
                patches = [Patch(color=c["color"], label=c["name"]) for c in SNOW_COVER_CLASSES]
                ax.legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.9)
                buf = BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                plt.close()
                buf.seek(0)
                st.image(Image.open(buf))

            with col_s:
                names = [s["name"] for s in result.stats]
                ratios = [s["ratio"] * 100 for s in result.stats]
                colors = [s["color"] for s in result.stats]

                fig2, ax2 = plt.subplots(figsize=(5, 4))
                bars = ax2.barh(names, ratios, color=colors, edgecolor="#333")
                ax2.set_xlabel("面积占比 (%)", fontsize=10)
                ax2.set_title("积雪覆盖分布", fontsize=12)
                for bar, val in zip(bars, ratios):
                    if val > 2:
                        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                                f"{val:.1f}%", va="center", fontsize=9)
                plt.tight_layout()
                buf2 = BytesIO()
                plt.savefig(buf2, format="png", dpi=100, bbox_inches="tight")
                plt.close()
                buf2.seek(0)
                st.image(Image.open(buf2))

            with st.expander("📋 详细统计表"):
                df_snow = pd.DataFrame(result.stats)
                df_snow = df_snow.rename(columns={
                    "name": "类型", "pixel_count": "像元数",
                    "ratio": "占比", "area_km2": "面积(km²)",
                })
                st.dataframe(
                    df_snow[["类型", "像元数", "占比", "面积(km²)"]],
                    height=300, hide_index=True,
                )

    # Tab 3: 冰川边界
    if use_glacier and result.glacier_mask is not None:
        with tabs[tab_idx]:
            tab_idx += 1
            st.caption("冰川边界提取 — NIR/SWIR 比值 + NDSI + NDVI 三重过滤")

            # 冰川分类 RGB
            glacier_rgb = np.zeros((*result.glacier_mask.shape, 3), dtype=np.uint8)
            for cls in GLACIER_CLASSES:
                rgb = tuple(int(c * 255) for c in to_rgb(cls["color"]))
                glacier_rgb[result.glacier_mask == cls["code"]] = rgb

            col_m, col_s = st.columns([3, 2])
            with col_m:
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(glacier_rgb)
                ax.set_title(f"冰川边界 — {area_name}", fontsize=13)
                ax.axis("off")
                patches = [Patch(color=c["color"], label=c["name"]) for c in GLACIER_CLASSES]
                ax.legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.9)
                buf = BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                plt.close()
                buf.seek(0)
                st.image(Image.open(buf))

            with col_s:
                glacier_stats = compute_glacier_stats(result.glacier_mask, pixel_size)
                for g in glacier_stats:
                    st.metric(
                        f"{g['name']}",
                        f"{g['area_km2']:.2f} km²",
                    )
                glacier_ratio = sum(g["ratio"] for g in glacier_stats if g["code"] > 0)
                st.metric("冰川总面积占比", f"{glacier_ratio*100:.1f}%")

            # 雪线信息
            with st.expander("🏔️ 雪线估计"):
                sl = s["snow_line"]
                if sl["available"]:
                    st.metric("估计雪线高程", f"{sl['snow_line_elevation']} m")
                else:
                    st.info("需要 DEM 数据才能计算雪线高度。当前仅给出云雪覆盖比例。")
                st.metric("积雪覆盖率", f"{sl['snow_cover_ratio']*100:.1f}%")
                st.caption(f"方法: {sl['method']}")

    # Tab 4: 冻土分析
    if use_frozen:
        with tabs[tab_idx]:
            st.caption("基于 NDVI + NDSI 的冻土活动层状态推断")

            fg = s["frozen_ground"]
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                st.metric("❄️ 冻土比例", f"{fg['frozen_ratio']*100:.1f}%")
                st.caption(f"{fg['frozen_pixels']:,} 像元")
            with col_f2:
                st.metric("💧 融化中比例", f"{fg['thawing_ratio']*100:.1f}%")
                st.caption(f"{fg['thawing_pixels']:,} 像元")
            with col_f3:
                st.metric("🌿 完全融化比例", f"{fg['active_ratio']*100:.1f}%")
                st.caption(f"{fg['active_pixels']:,} 像元")

            st.info(
                f"**冻土状态**: {fg['state']}\n\n"
                "💡 冻土分析基于积雪覆盖 + 植被状态间接推断。精确冻土活动层厚度需要 "
                "地表温度 (LST) + 土壤水分 + 钻孔数据。"
            )

    # ---- 多时相雪盖变化 ----
    if use_timeline and len(all_ndsi) >= 2:
        st.divider()
        st.subheader("📈 多时相雪盖变化")
        cols_t = st.columns(min(4, len(all_ndsi)))
        for i, (ndsi_i, label) in enumerate(zip(all_ndsi, image_labels)):
            with cols_t[i % 4]:
                snow_ratio = (ndsi_i > ndsi_threshold).sum() / max(ndsi_i.size, 1)
                fig_t, ax_t = plt.subplots(figsize=(3, 3))
                ax_t.imshow(ndsi_i, cmap="Blues_r", vmin=-0.3, vmax=0.9)
                ax_t.set_title(f"{label}\n雪盖 {snow_ratio*100:.1f}%", fontsize=9)
                ax_t.axis("off")
                buf_t = BytesIO()
                plt.tight_layout()
                plt.savefig(buf_t, format="png", dpi=80, bbox_inches="tight")
                plt.close()
                buf_t.seek(0)
                st.image(Image.open(buf_t))

    # ---- 导出 ----
    st.divider()
    st.subheader("📥 结果导出")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        all_rows = []
        for stat in result.stats:
            all_rows.append({
                "类型": stat["name"],
                "像元数": stat["pixel_count"],
                "占比": stat["ratio"],
                "面积_km2": stat["area_km2"],
            })
        df_export = pd.DataFrame(all_rows)
        csv = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📊 下载积雪分类统计 CSV",
            data=csv,
            file_name=f"cryosphere_{area_name}_{main_date}.csv",
            mime="text/csv",
            width="stretch",
        )
    with col_d2:
        st.info(
            "💡 **提示**: 完整 GeoTIFF 导出请使用「报告导出」页面。\n\n"
            "雪线精确高程需要 **DEM 辅助** (SRTM 30m / ASTER GDEM)。\n"
            "冻土活动层厚度需要 **LST + 土壤水分数据**。"
        )

else:
    st.info("👈 在左侧设置参数，点击「搜索影像 & 分析」开始冰冻圈分析")

    with st.expander("📖 冰冻圈分析说明"):
        st.markdown("""
        ### 冰冻圈遥感指标一览

        | 指标 | 全称 | 公式 | 判据 |
        |------|------|------|------|
        | **NDSI** | 归一化积雪指数 | (Green - SWIR1)/(Green + SWIR1) | >0.4 积雪 |
        | **Snow Cover** | 积雪覆盖 | NDSI 4级分类 | 裸地/薄雪/积雪/冰川 |
        | **Glacier** | 冰川边界 | NIR/SWIR+NDSI+NDVI | 三重过滤 |
        | **Snow Line** | 雪线高度 | NDSI+DEM | 需要 DEM |
        | **Frozen** | 冻土状态 | NDVI+NDSI | 间接推断 |

        ### 推荐研究区
        - 🏔️ **天山北坡** → 冰川融水补给绿洲
        - 🏔️ **柴达木盆地** → 周边高山冰川发育
        - 🏔️ **祁连山/河西走廊** → 冰川-绿洲系统

        ### 使用流程
        1. 选择有冰川发育的研究区 (天山/祁连山)
        2. 选择夏季晴空影像 (低云量)
        3. 勾选分析模块
        4. 查看 NDSI/雪盖/冰川结果
        """)
