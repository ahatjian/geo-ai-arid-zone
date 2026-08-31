"""
研究区地图渲染工具 — leafmap 失败时自动降级为静态地图
======================================================
leafmap/Esri 底图在国内网络可能加载失败导致空白。
本模块提供:
  - render_area_map(): 单研究区地图 (leafmap 优先, matplotlib 降级)
  - draw_study_areas_overview(): 6 大研究区概览示意图 (matplotlib)

降级静态地图绘制: bbox 矩形 + 研究区中心点 + 经纬度网格,
不依赖外部底图, 保证任何网络下都有图可看。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def draw_area_schematic(
    area_name: str,
    bbox: List[float],
    center: Optional[List[float]] = None,
    all_areas: Optional[Dict] = None,
    figsize: Tuple[int, int] = (8, 5),
) -> Optional[bytes]:
    """
    用 matplotlib 绘制研究区示意图 (静态, 无外部依赖)。

    绘制: 中国西北区域轮廓示意 + 研究区 bbox + 中心点 + 名称。

    参数:
        area_name: 研究区名
        bbox: [min_lon, min_lat, max_lon, max_lat]
        center: 中心点 [lat, lon] (None=bbox 计算)
        all_areas: 全部研究区 (用于对比显示, 可选)
        figsize: 图像尺寸

    返回:
        bytes: PNG 字节 (失败返回 None)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        return None

    min_lon, min_lat, max_lon, max_lat = bbox
    if center is None:
        center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]

    fig, ax = plt.subplots(figsize=figsize)

    # 背景: 西北干旱区示意范围
    ax.set_xlim(73, 110)
    ax.set_ylim(33, 50)
    ax.set_facecolor("#f5f0e8")  # 沙漠色背景

    # 经纬度网格
    ax.set_xticks(range(75, 111, 5))
    ax.set_yticks(range(35, 51, 5))
    ax.grid(True, linestyle="--", alpha=0.4, color="#c8b8a8")
    ax.set_xlabel("经度 (°E)")
    ax.set_ylabel("纬度 (°N)")
    ax.set_title(f"📍 {area_name} 研究区位置", fontsize=13, fontweight="bold")

    # 其他研究区 (浅色)
    if all_areas:
        for name, info in all_areas.items():
            if name == area_name:
                continue
            b = info["bbox"]
            rect = Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                             facecolor="#d4c4a8", edgecolor="#a89a80",
                             linewidth=1, alpha=0.6)
            ax.add_patch(rect)
            ax.text((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, name,
                    ha="center", va="center", fontsize=7, color="#7a6a50")

    # 当前研究区 (高亮)
    rect = Rectangle((min_lon, min_lat), max_lon - min_lon, max_lat - min_lat,
                     facecolor="#1f77b4", edgecolor="#e74c3c",
                     linewidth=2, alpha=0.35)
    ax.add_patch(rect)
    ax.plot(center[1], center[0], "o", color="#e74c3c", markersize=6)
    ax.annotate(area_name, (center[1], center[0]),
                textcoords="offset points", xytext=(8, 8),
                fontsize=10, fontweight="bold", color="#1a5276")

    # 边界坐标标注
    ax.text(min_lon, max_lat + 0.4, f"{min_lon:.1f}°~{max_lon:.1f}°E",
            fontsize=8, ha="left", color="#666666")
    ax.text(min_lon - 0.3, (min_lat + max_lat) / 2, f"{min_lat:.1f}°~{max_lat:.1f}°N",
            fontsize=8, va="center", rotation=90, color="#666666")

    plt.tight_layout()
    buf = __import__("io").BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_area_map(
    area_name: str,
    bbox: List[float],
    center: Optional[List[float]] = None,
    all_areas: Optional[Dict] = None,
    height: int = 350,
) -> Tuple[str, Optional[bytes]]:
    """
    渲染研究区地图: leafmap 优先, 失败自动降级为静态图。

    返回:
        (mode, image_bytes)
        - mode: "leafmap" (已渲染交互地图, image=None)
                | "static" (返回静态 PNG) | "error" (均失败)
    """
    # 尝试 leafmap 交互地图
    try:
        import streamlit as st
        import leafmap
        from shapely.geometry import box
        import geopandas as gpd

        m = leafmap.Map(center=center if center else [39.5, 87], zoom=5, height=height)
        m.add_basemap("Esri.WorldImagery")
        bbox_geom = box(*bbox)
        gdf = gpd.GeoDataFrame(
            {"name": [area_name]}, geometry=[bbox_geom], crs="EPSG:4326"
        )
        m.add_gdf(gdf, layer_name=area_name,
                  style={"color": "red", "fillOpacity": 0.05, "weight": 2})
        m.to_streamlit(height=height)
        return "leafmap", None
    except Exception:
        pass

    # 降级: matplotlib 静态图
    img = draw_area_schematic(area_name, bbox, center, all_areas)
    if img:
        return "static", img
    return "error", None
