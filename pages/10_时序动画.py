"""
时序动画生成页面 — NDVI/水体/雪盖年际变化 GIF
==============================================
支持单指数动画 / 多指数并列动画 / 趋势曲线动画
"""
import streamlit as st
import os, sys, tempfile, numpy as np
from datetime import date, timedelta
from io import BytesIO
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import STUDY_AREAS, COLLECTIONS
from utils.error_handler import StreamlitErrorBoundary
from utils.pc_data import search_images, get_rgb_preview_cached, download_multiband

st.set_page_config(page_title="时序动画", page_icon="🎬", layout="wide")

with st.sidebar:
    st.title("🎬 动画设置")
    default_area = st.session_state.get("selected_area", "塔里木盆地")
    if default_area not in STUDY_AREAS:
        default_area = "塔里木盆地"
    area_name = st.selectbox("研究区", list(STUDY_AREAS.keys()),
                            index=list(STUDY_AREAS.keys()).index(default_area))
    area_info = STUDY_AREAS[area_name]
    st.session_state["selected_area"] = area_name
    st.caption(area_info["description"])

    st.divider()
    satellite = st.selectbox("数据源", list(COLLECTIONS.keys()),
                            format_func=lambda x: f"{x} ({COLLECTIONS[x]['resolution']}m)")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始", date.today() - timedelta(days=365))
    with col2:
        end_date = st.date_input("结束", date.today())

    cloud_cover = st.slider("最大云量 (%)", 0, 100, 20)
    max_items = st.slider("最大影像数", 3, 24, 12, help="越多影像 = 越流畅的动画")

    st.divider()
    anim_type = st.selectbox("动画类型", ["NDVI 植被变化", "MNDWI 水体变化", "NDSI 雪盖变化", "多指数对比", "趋势曲线"])

    with st.expander("⚙️ 动画参数"):
        fps = st.slider("帧率 (fps)", 1, 10, 2)
        dpi = st.slider("分辨率 (dpi)", 60, 200, 100)

    search_clicked = st.button("🔍 搜索 & 生成动画", type="primary")

# === 主页面 ===
st.title("🎬 时序遥感动画")
st.markdown(f"**{area_name}** | {satellite} | {start_date} → {end_date}")

if search_clicked:
    bbox = area_info["bbox"]

    with st.spinner("🔍 搜索影像..."):
        try:
            results = search_images(bbox=bbox, start_date=start_date.strftime("%Y-%m-%d"),
                                    end_date=end_date.strftime("%Y-%m-%d"),
                                    collection=satellite, cloud_cover_max=cloud_cover,
                                    max_items=max_items)
        except Exception as e:
            st.error(f"搜索失败: {e}")
            results = []

    if not results:
        st.warning("⚠️ 未找到影像")
        st.stop()

    st.success(f"✅ {len(results)} 景影像")

    # 下载所有影像
    progress = st.progress(0)
    status = st.empty()
    all_ndvi, all_mndwi, all_ndsi = [], [], []
    image_labels = []

    for idx, r in enumerate(results):
        pct = (idx + 1) / len(results)
        progress.progress(pct)
        status.text(f"⬇️ [{idx+1}/{len(results)}] {r['datetime']}")

        try:
            tmp = os.path.join(tempfile.gettempdir(), f"anim_{r['id'][:12]}.tif")
            tif = download_multiband(r["item"], tmp, collection=satellite)
            if tif and os.path.exists(tif):
                import rasterio
                with rasterio.open(tif) as src:
                    b = src.read().astype(np.float64)
                if np.nanmedian(b[1]) > 10:
                    b /= 10000.0

                blue, green, red, nir, swir1 = b[0], b[1], b[2], b[3], b[4]

                # NDVI
                d = nir + red
                ndvi = np.where(d > 1e-6, (nir - red) / d, 0)
                ndvi = np.clip(ndvi, -1, 1)
                all_ndvi.append(ndvi.astype(np.float32))

                # MNDWI
                d2 = green + swir1
                mndwi = np.where(d2 > 1e-6, (green - swir1) / d2, 0)
                mndwi = np.clip(mndwi, -1, 1)
                all_mndwi.append(mndwi.astype(np.float32))

                # NDSI
                d3 = green + swir1
                ndsi = np.where(d3 > 1e-6, (green - swir1) / d3, 0)
                ndsi = np.clip(ndsi, -1, 1)
                all_ndsi.append(ndsi.astype(np.float32))

                image_labels.append(r["datetime"])
                try:
                    os.remove(tif)
                except Exception:
                    pass
        except Exception as e:
            st.warning(f"⚠️ {r['datetime']}: {e}")

    progress.progress(1.0)
    status.text("✅ 下载完成")

    if len(all_ndvi) < 2:
        st.error("❌ 有效影像不足")
        st.stop()

    st.success(f"✅ {len(all_ndvi)} 景用于动画")

    # 生成动画
    from utils.animation import create_timeseries_animation, create_multi_index_animation, create_trend_animation

    with st.spinner("🎬 渲染动画中..."):
        ndvi_stack = np.stack(all_ndvi, axis=-1)

        if anim_type == "NDVI 植被变化":
            gif = create_timeseries_animation(ndvi_stack, image_labels,
                                              cmap="RdYlGn", vmin=-0.2, vmax=0.9,
                                              title=f"{area_name} NDVI 变化",
                                              label="NDVI", fps=fps, dpi=dpi)
        elif anim_type == "MNDWI 水体变化":
            mndwi_stack = np.stack(all_mndwi, axis=-1)
            gif = create_timeseries_animation(mndwi_stack, image_labels,
                                              cmap="Blues", vmin=-0.5, vmax=0.5,
                                              title=f"{area_name} MNDWI 水体变化",
                                              label="MNDWI", fps=fps, dpi=dpi)
        elif anim_type == "NDSI 雪盖变化":
            ndsi_stack = np.stack(all_ndsi, axis=-1)
            gif = create_timeseries_animation(ndsi_stack, image_labels,
                                              cmap="Blues_r", vmin=-0.3, vmax=0.9,
                                              title=f"{area_name} NDSI 雪盖变化",
                                              label="NDSI", fps=fps, dpi=dpi)
        elif anim_type == "多指数对比":
            data = {}
            if all_ndvi:
                data["NDVI"] = ndvi_stack
            if all_mndwi:
                data["MNDWI"] = np.stack(all_mndwi, axis=-1)
            if all_ndsi:
                data["NDSI"] = np.stack(all_ndsi, axis=-1)
            gif = create_multi_index_animation(data, image_labels,
                                               title=f"{area_name} 多指数对比", fps=fps, dpi=dpi)
        else:  # 趋势曲线
            means = [np.nanmean(f) for f in all_ndvi]
            gif = create_trend_animation(np.array(means), image_labels,
                                        title=f"{area_name} NDVI 趋势", ylabel="NDVI")

    if gif:
        st.subheader("🎬 生成的动画")
        st.image(gif, caption=f"{len(all_ndvi)} 帧 @ {fps} fps")
        st.download_button("💾 下载 GIF", gif,
                          f"animation_{area_name}_{anim_type}.gif",
                          "image/gif")
    else:
        st.error("❌ 动画生成失败")

else:
    st.info("👈 选择研究区和时间范围，点击搜索开始生成动画")

    with st.expander("📖 使用说明"):
        st.markdown("""
        ### 动画类型说明

        | 类型 | 用途 | 推荐研究区 |
        |------|------|-----------|
        | NDVI 植被变化 | 绿洲扩张/退化年际变化 | 塔里木/河西走廊 |
        | MNDWI 水体变化 | 湖泊面积季节波动 | 柴达木/艾丁湖 |
        | NDSI 雪盖变化 | 积雪消融过程 | 天山北坡 |
        | 多指数对比 | 植被+水体+雪盖同屏 | 任意 |
        | 趋势曲线 | NDVI 均值动态走势 | 任意 |

        💡 选择 **12+ 景** 影像可获得流畅动画效果
        """)
