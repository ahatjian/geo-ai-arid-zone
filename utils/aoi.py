"""
自定义研究区 (AOI) 选择组件
==============================
提供统一的研究区/AOI 选择能力，供所有分析页面复用。

功能:
  - 预设研究区选择 (6 个干旱区)
  - 自定义 AOI: 手动输入经纬度 bbox
  - 自定义 AOI: 上传 GeoJSON 矢量边界 (自动解析 bbox)
  - 统一写入 session_state (selected_bbox / selected_center / selected_area)

依赖: streamlit, shapely (解析 GeoJSON)
"""

import json
from typing import Optional, Tuple, List, Dict

import streamlit as st

from config import STUDY_AREAS


def parse_geojson_bbox(content_bytes: bytes) -> Tuple[List[float], str]:
    """
    解析 GeoJSON 内容，返回 (bbox, name)

    支持 FeatureCollection / Feature / Geometry 三种 GeoJSON 结构，
    多个几何取并集后返回其外接矩形 bbox。

    参数:
        content_bytes: GeoJSON 文件字节内容

    返回:
        (bbox, name) — bbox 为 [min_lon, min_lat, max_lon, max_lat]
    """
    from shapely.geometry import shape
    from shapely.ops import unary_union

    data = json.loads(content_bytes.decode("utf-8"))

    geoms = []
    gtype = data.get("type")

    if gtype == "FeatureCollection":
        for feat in data.get("features", []):
            geoms.append(shape(feat["geometry"]))
    elif gtype == "Feature":
        geoms.append(shape(data["geometry"]))
    else:
        # 直接是 Geometry 对象
        geoms.append(shape(data))

    if not geoms:
        raise ValueError("GeoJSON 中未找到任何几何对象")

    union = unary_union(geoms)
    minx, miny, maxx, maxy = union.bounds

    return [minx, miny, maxx, maxy], "自定义AOI(GeoJSON)"


def validate_bbox(bbox: List[float]) -> bool:
    """验证 bbox 是否合法"""
    if len(bbox) != 4:
        return False
    lon_min, lat_min, lon_max, lat_max = bbox
    if lon_min >= lon_max or lat_min >= lat_max:
        return False
    if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
        return False
    if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
        return False
    return True


def render_aoi_selector(
    default_area: str = "塔里木盆地",
    help_text: str = "",
    key_prefix: str = "",
) -> Tuple[Optional[str], Optional[List[float]], Optional[List[float]], str, Optional[Dict]]:
    """
    渲染研究区 / AOI 选择组件 (放在侧边栏中)

    参数:
        default_area: 默认预设研究区名
        help_text: 研究区选择帮助文本
        key_prefix: widget key 前缀 (避免多页面冲突)

    返回:
        (name, bbox, center, source, info)
        - name: 研究区/AOI 名称 (自定义无效时为 None)
        - bbox: [min_lon, min_lat, max_lon, max_lat] (无效时为 None)
        - center: [lat, lon] (无效时为 None)
        - source: "preset" | "custom"
        - info: 研究区信息 dict (含 bbox/center/description)
    """
    st.subheader("研究区")

    mode = st.radio(
        "选择方式",
        ["预设研究区", "自定义 AOI"],
        horizontal=True,
        key=f"{key_prefix}_aoi_mode",
    )

    if mode == "预设研究区":
        default = st.session_state.get("selected_area", default_area)
        if default not in STUDY_AREAS:
            default = default_area
            if default not in STUDY_AREAS:
                default = list(STUDY_AREAS.keys())[0]

        area_name = st.selectbox(
            "研究区",
            list(STUDY_AREAS.keys()),
            index=list(STUDY_AREAS.keys()).index(default),
            help=help_text,
            key=f"{key_prefix}_area_select",
        )
        info = STUDY_AREAS[area_name]
        bbox = info["bbox"]

        st.session_state["selected_area"] = area_name
        st.session_state["selected_bbox"] = bbox
        st.session_state["selected_center"] = info["center"]

        st.caption(f"📌 {info['description']}")
        return area_name, bbox, info["center"], "preset", info

    else:
        # ---- 自定义 AOI ----
        st.caption("输入经纬度范围，或上传 GeoJSON 边界")

        # 从 session 恢复上次的自定义 bbox
        prev_bbox = st.session_state.get("custom_bbox", None)

        col1, col2 = st.columns(2)
        with col1:
            lon_min = st.number_input(
                "最小经度 (西)", value=float(prev_bbox[0]) if prev_bbox else 80.0,
                format="%.3f", key=f"{key_prefix}_lon_min",
            )
            lat_min = st.number_input(
                "最小纬度 (南)", value=float(prev_bbox[1]) if prev_bbox else 38.0,
                format="%.3f", key=f"{key_prefix}_lat_min",
            )
        with col2:
            lon_max = st.number_input(
                "最大经度 (东)", value=float(prev_bbox[2]) if prev_bbox else 90.0,
                format="%.3f", key=f"{key_prefix}_lon_max",
            )
            lat_max = st.number_input(
                "最大纬度 (北)", value=float(prev_bbox[3]) if prev_bbox else 44.0,
                format="%.3f", key=f"{key_prefix}_lat_max",
            )

        bbox = [lon_min, lat_min, lon_max, lat_max]
        name = "自定义AOI"
        desc = "自定义研究区"

        # GeoJSON 上传 (覆盖手动 bbox)
        geojson_file = st.file_uploader(
            "或上传 GeoJSON 边界 (可选，覆盖上方 bbox)",
            type=["geojson", "json"],
            key=f"{key_prefix}_geojson",
        )
        if geojson_file is not None:
            try:
                bbox, name = parse_geojson_bbox(geojson_file.getvalue())
                desc = f"GeoJSON 边界: {name}"
                st.success(
                    f"✅ 已解析 GeoJSON\n\n"
                    f"bbox: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]"
                )
            except Exception as e:
                st.error(f"❌ GeoJSON 解析失败: {e}")

        # 验证
        if not validate_bbox(bbox):
            st.error("⚠️ bbox 无效：最小经纬度必须小于最大经纬度，且经度∈[-180,180]、纬度∈[-90,90]")
            return None, None, None, "custom", None

        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        st.session_state["custom_bbox"] = bbox
        st.session_state["selected_bbox"] = bbox
        st.session_state["selected_center"] = center

        info = {"bbox": bbox, "center": center, "description": desc, "keywords": []}
        st.caption(f"📌 {desc} — 中心 [{center[0]:.2f}, {center[1]:.2f}]")

        return name, bbox, center, "custom", info
