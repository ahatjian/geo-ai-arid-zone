"""
农业干旱分析页面 — CWSI / SMI / 灌溉需求
==========================================
基于 Sentinel-2/Landsat 的农田水分监测
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
from utils.pc_data import (
    search_images, get_rgb_preview_cached, download_multiband,
)

st.set_page_config(page_title="农业干旱", page_icon="🌾", layout="wide")

st.markdown("""
<style>
.agri-card {
    background: #1a1a2e; border: 1px solid #333;
    border-radius: 8px; padding: 14px 18px; text-align: center;
}
.agri-card .label { font-size: 11px; color: #f0c040; margin-bottom: 4px; }
.agri-card .value { font-size: 20px; font-weight: 700; color: #e8eaed; }
.priority-urgent { background: linear-gradient(135deg, #e74c3c22, #e74c3c11); border-color: #e74c3c !important; }
.priority-high { background: linear-gradient(135deg, #e67e2222, #e67e2211); border-color: #e67e22 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.title("🌾 农业干旱设置")

    st.subheader("研究区")
    default_area = st.session_state.get("selected_area", "河西走廊")
    if default_area not in STUDY_AREAS:
        default_area = "河西走廊"
    area_name = st.selectbox(
        "选择研究区", list(STUDY_AREAS.keys()),
        index=list(STUDY_AREAS.keys()).index(default_area),
        help="推荐河西走廊/天山北坡 (绿洲农业区)",
    )
    area_info = STUDY_AREAS[area_name]
    st.session_state["selected_area"] = area_name
    st.caption(f"📌 {area_info['description']}")

    st.divider()

    st.subheader("数据源")
    satellite = st.selectbox(
        "卫星数据", list(COLLECTIONS.keys()),
        format_func=lambda x: f"{x} ({COLLECTIONS[x]['resolution']}m)",
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", date.today() - timedelta(days=120))
    with col2:
        end_date = st.date_input("结束日期", date.today())

    cloud_cover = st.slider("最大云量 (%)", 0, 100, 10)
    max_items = st.slider("最大影像数", 1, 10, 3)

    st.divider()

    st.subheader("作物类型")
    crop_type = st.selectbox(
        "作物", ["general", "wheat", "corn", "cotton"],
        format_func=lambda x: {"general": "通用", "wheat": "小麦", "corn": "玉米", "cotton": "棉花"}[x],
    )

    st.divider()

    st.subheader("分析选项")
    use_cwsi = st.checkbox("CWSI 作物水分胁迫", value=True)
    use_smi = st.checkbox("SMI 土壤水分指数", value=True)
    use_irrigation = st.checkbox("灌溉需求评估", value=True)

    with st.expander("⚙️ 高级设置"):
        pixel_size = st.number_input("像元大小 (m)", value=10.0, min_value=1.0)
        cwsi_wet = st.number_input("CWIS 湿润参考 NDVI", value=0.0, min_value=0.0, max_value=1.0,
                                   help="0=自动 (建议)")
        cwsi_dry = st.number_input("CWIS 干旱参考 NDVI", value=0.0, min_value=0.0, max_value=1.0,
                                   help="0=自动 (建议)")

    search_clicked = st.button("🔍 搜索影像 & 分析", type="primary", use_container_width=True)

# ============================================================
# 主页面
# ============================================================
st.title("🌾 农业干旱遥感分析")
st.markdown(f"**研究区: {area_name}** | {satellite} | {start_date} → {end_date}")

if search_clicked:
    bbox = area_info["bbox"]

    with st.spinner("🔍 搜索影像..."):
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
        st.warning("⚠️ 未找到影像")
        st.stop()

    st.success(f"✅ 找到 {len(results)} 景影像")

    # 影像选择
    cols = st.columns(3)
    selected_items = []
    for i, r in enumerate(results):
        with cols[i % 3]:
            try:
                preview = get_rgb_preview_cached(r["id"], collection=satellite, width=256)
            except Exception:
                preview = None
            if preview:
                st.image(preview, use_container_width=True)
            else:
                st.markdown('<div style="height:100px;background:#1a1a2e;border-radius:6px;display:flex;'
                           'align-items:center;justify-content:center;color:#666">无预览</div>',
                           unsafe_allow_html=True)
            selected = st.checkbox(f"**{r['datetime']}** ☁️{r['cloud_cover']}%",
                                   key=f"sel_agri_{r['id']}", value=i == 0)
            if selected:
                selected_items.append(r)

    if not selected_items:
        st.warning("⚠️ 请选择至少 1 景影像")
        st.stop()

    # 下载分析
    progress_bar = st.progress(0)
    status_text = st.empty()
    all_bands = []
    image_labels = []

    for idx, item in enumerate(selected_items):
        pct = (idx + 1) / len(selected_items)
        progress_bar.progress(pct)
        status_text.text(f"⬇️ [{idx+1}/{len(selected_items)}] {item['datetime']}")

        try:
            tmp_path = os.path.join(tempfile.gettempdir(), f"agri_{item['id'][:12]}.tif")
            tif_path = download_multiband(item["item"], tmp_path, collection=satellite)
            if tif_path and os.path.exists(tif_path):
                import rasterio
                with rasterio.open(tif_path) as src:
                    bands_data = src.read().astype(np.float32)
                all_bands.append(bands_data)
                image_labels.append(item["datetime"])
                try:
                    os.remove(tif_path)
                except Exception:
                    pass
        except Exception as e:
            st.warning(f"⚠️ {item['datetime']}: {e}")

    progress_bar.progress(1.0)
    status_text.text("✅ 完成")

    if not all_bands:
        st.error("❌ 无有效影像")
        st.stop()

    bands_main = all_bands[0]
    main_date = image_labels[0]

    # 分析
    from utils.agri_drought import (
        assess_agri_drought, calc_cwsi_ndvi, calc_smi_swir,
        calc_smi_combined, calc_mpdi, classify_agri_drought,
        estimate_irrigation_demand, AGRI_DROUGHT_LEVELS,
    )

    ndvi_wet_arg = cwsi_wet if cwsi_wet > 0 else None
    ndvi_dry_arg = cwsi_dry if cwsi_dry > 0 else None

    with st.spinner("⏳ 农业干旱评估..."):
        red = bands_main[2].astype(np.float64)
        nir = bands_main[3].astype(np.float64)
        swir1 = bands_main[4].astype(np.float64)

        denom = nir + red
        ndvi = np.where(denom > 1e-6, (nir - red) / denom, 0.0)
        ndvi = np.clip(ndvi, -1.0, 1.0)

        if ndvi_wet_arg:
            cwsi = calc_cwsi_ndvi(ndvi, ndvi_wet=ndvi_wet_arg, ndvi_dry=ndvi_dry_arg)
        else:
            cwsi = calc_cwsi_ndvi(ndvi)

        smi = calc_smi_swir(swir1)
        smi_comb = calc_smi_combined(ndvi, swir1)
        mpdi = calc_mpdi(red, nir)

        category = classify_agri_drought(cwsi, smi=smi, method="composite")
        irrigation = estimate_irrigation_demand(cwsi, smi=smi, crop_type=crop_type)

        from utils.agri_drought import compute_agri_stats
        stats = compute_agri_stats(category, pixel_size)

    # ---- 汇总 ----
    st.subheader("📊 农业干旱评估")
    priority_css = "priority-urgent" if irrigation["irrigation_priority"] == "紧急" else ("priority-high" if irrigation["irrigation_priority"] == "高" else "")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="agri-card {priority_css}"><div class="label">CWSI 均值</div><div class="value">{irrigation["mean_cwsi"]:.3f}</div></div>', unsafe_allow_html=True)
    with col2:
        mod_ratio = sum(s["ratio"] for s in stats if s["code"] >= 2)
        st.markdown(f'<div class="agri-card"><div class="label">中度以上干旱</div><div class="value">{mod_ratio*100:.1f}%</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="agri-card"><div class="label">日需水量</div><div class="value">{irrigation["daily_water_demand_mm"]:.1f} mm</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="agri-card"><div class="label">灌溉优先级</div><div class="value">{irrigation["irrigation_priority"]}</div></div>', unsafe_allow_html=True)

    st.info(f"💧 **灌溉建议**: {irrigation['irrigation_advice']} | 紧急区占比: {irrigation['urgent_area_ratio']*100:.1f}% | 作物类型: {crop_type}")

    # ---- Tab ----
    tab_names = []
    if use_cwsi: tab_names.append("CWSI 水分胁迫")
    if use_smi: tab_names.append("SMI 土壤水分")
    if use_irrigation: tab_names.append("灌溉需求")

    if not tab_names:
        tab_names = ["CWSI 水分胁迫"]

    tabs = st.tabs(tab_names)
    t_idx = 0

    if use_cwsi:
        with tabs[t_idx]:
            t_idx += 1
            st.caption("CWSI: 0=无胁迫, 1=极度胁迫")

            col_m, col_s = st.columns([3, 2])
            with col_m:
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(cwsi, cmap="YlOrRd", vmin=0, vmax=1)
                ax.set_title(f"CWSI — {area_name} ({main_date})", fontsize=13)
                ax.axis("off")
                cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
                cbar.set_label("CWSI", fontsize=10)
                buf = BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                plt.close()
                buf.seek(0)
                st.image(Image.open(buf), use_container_width=True)

            with col_s:
                names = [s["name"] for s in stats]
                ratios = [s["ratio"] * 100 for s in stats]
                colors = [s["color"] for s in stats]

                fig2, ax2 = plt.subplots(figsize=(5, 4))
                ax2.barh(names, ratios, color=colors, edgecolor="#333")
                ax2.set_xlabel("面积占比 (%)", fontsize=10)
                ax2.set_title("农业干旱等级", fontsize=12)
                for bar, val in zip(ax2.containers[0], ratios):
                    if val > 3:
                        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                                f"{val:.1f}%", va="center", fontsize=9)
                plt.tight_layout()
                buf2 = BytesIO()
                plt.savefig(buf2, format="png", dpi=100, bbox_inches="tight")
                plt.close()
                buf2.seek(0)
                st.image(Image.open(buf2), use_container_width=True)

    if use_smi:
        with tabs[t_idx]:
            t_idx += 1
            st.caption("SMI: 0=极干, 1=饱和")

            col_m, col_s = st.columns([3, 1])
            with col_m:
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(smi_comb, cmap="Blues", vmin=0, vmax=1)
                ax.set_title(f"SMI 土壤水分 — {area_name}", fontsize=13)
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04).set_label("SMI", fontsize=10)
                buf = BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                plt.close()
                buf.seek(0)
                st.image(Image.open(buf), use_container_width=True)

            with col_s:
                smi_v = smi_comb[np.isfinite(smi_comb)]
                st.metric("均值", f"{np.nanmean(smi_v):.4f}")
                st.metric("标准差", f"{np.nanstd(smi_v):.4f}")
                dry_px = int((smi_comb < 0.2).sum())
                st.metric("干旱像元", f"{dry_px:,}")
                st.caption("SMI < 0.2 = 干旱")

    if use_irrigation:
        with tabs[t_idx]:
            st.caption("灌溉需求评估")

            # 需求分级图
            demand_rgb = np.zeros((*category.shape, 3), dtype=np.uint8)
            colors_req = {0: [46,204,113], 1: [241,196,15], 2: [230,126,34], 3: [231,76,60], 4: [142,68,173]}
            for code, rgb in colors_req.items():
                demand_rgb[category == code] = rgb

            col_m, col_s = st.columns([3, 2])
            with col_m:
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(demand_rgb)
                ax.set_title(f"灌溉需求分区 — {area_name}", fontsize=13)
                ax.axis("off")
                from matplotlib.patches import Patch
                patches = [Patch(color=f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}",
                                label=AGRI_DROUGHT_LEVELS[i]["name"]) for i, c in colors_req.items()]
                ax.legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.9)
                buf = BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                plt.close()
                buf.seek(0)
                st.image(Image.open(buf), use_container_width=True)

            with col_s:
                for stat in stats:
                    st.metric(f"{stat['name']}", f"{stat['ratio']*100:.1f}%")
                    st.caption(stat["irrigation"])

            # 灌溉建议详情
            with st.expander("💧 灌溉详细建议"):
                st.markdown(f"""
                **灌溉优先级**: {irrigation['irrigation_priority']}

                **日需水量估算**: {irrigation['daily_water_demand_mm']:.1f} mm/天

                **紧急灌溉区占比**: {irrigation['urgent_area_ratio']*100:.1f}%

                **建议**: {irrigation['irrigation_advice']}

                ---
                💡 以上为遥感估算值，实际灌溉量需结合土壤类型、气象数据校准。
                """)

    # ---- 导出 ----
    st.divider()
    st.subheader("📥 导出")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        df_export = pd.DataFrame(stats)
        csv = df_export.to_csv(index=False).encode("utf-8")
        st.download_button("📊 下载农业干旱统计 CSV", csv,
                          f"agri_drought_{area_name}_{main_date}.csv",
                          "text/csv", use_container_width=True)
    with col_d2:
        st.info("💡 精确灌溉量需结合 **气象站数据** (降水/蒸发) 和 **土壤类型** 校准。")

else:
    st.info("👈 选择绿洲农业研究区 (河西走廊/天山北坡)，点击搜索开始分析")

    with st.expander("📖 农业干旱指数说明"):
        st.markdown("""
        ### 农业干旱遥感指标

        | 指标 | 公式 | 用途 |
        |------|------|------|
        | **CWSI** | 1 - (NDVI - NDVI_dry)/(NDVI_wet - NDVI_dry) | 作物水分胁迫 |
        | **SMI** | (SWIR_max - SWIR)/(SWIR_max - SWIR_min) | 土壤水分含量 |
        | **MPDI** | (NIR + M*Red)/√(1+M²) | 植被-土壤水分综合 |
        | **Irrigation** | 基于 CWSI 分级 | 灌溉需求分区 |

        ### 农业干旱等级 (5级)
        | 等级 | CWSI | 建议 |
        |------|------|------|
        | 无干旱 | <0.2 | 无需灌溉 |
        | 轻度 | 0.2-0.4 | 建议灌溉 |
        | 中度 | 0.4-0.6 | 需要灌溉 |
        | 重度 | 0.6-0.8 | 急需灌溉 |
        | 极度 | >0.8 | 紧急灌溉 |

        ### 推荐研究区
        - 🌾 **河西走廊** → 绿洲农业走廊
        - 🌾 **天山北坡** → 冰川融水灌溉农业
        """)
