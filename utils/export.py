"""
Geo AI 结果导出模块 — 统一的导出入口

支持格式:
    - GeoTIFF: 栅格指数/掩膜/分类图
    - CSV: 统计表/像素计数/趋势数据
    - PNG/SVG: matplotlib 图表
    - GeoJSON: 矢量边界/变化区域多边形
    - NPY: numpy 数组中间结果

设计原则:
    - 所有导出函数返回 (文件路径, 二进制数据) 以便 Streamlit download_button 使用
    - 自动设置合理的默认参数
    - 统一的错误处理和日志
"""

import os
import io
import json
import tempfile
import warnings
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================
# 常量
# ============================================

INDEX_COLORMAPS = {
    "ndvi": "RdYlGn",
    "mndwi": "Blues",
    "awei": "Blues",
    "evi": "YlGn",
}

COMPRESSION = "lzw"
DEFAULT_NODATA = -9999


# ============================================
# GeoTIFF 导出
# ============================================

def export_raster_geotiff(
    data: np.ndarray,
    reference_path: str,
    output_path: Optional[str] = None,
    dtype: Optional[str] = None,
    nodata: Optional[float] = None,
    band_names: Optional[List[str]] = None,
    compress: str = COMPRESSION,
) -> Tuple[str, bytes]:
    """
    将 numpy 数组导出为 GeoTIFF，继承参考影像的空间参考。

    参数:
        data: 2D (单波段) 或 3D (多波段, bands×H×W) 数组
        reference_path: 参考 GeoTIFF 路径（CRS + transform）
        output_path: 输出路径（默认自动生成临时文件）
        dtype: 输出数据类型（默认继承 data.dtype）
        nodata: NoData 值
        band_names: 波段名称列表
        compress: 压缩方式

    返回:
        (文件路径, 文件二进制内容)
    """
    import rasterio

    if output_path is None:
        suffix = ".tif"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        output_path = tmp.name
        tmp.close()

    # 处理维度
    if data.ndim == 2:
        data = data[np.newaxis, ...]  # 1×H×W
    elif data.ndim != 3:
        raise ValueError(f"data 必须是 2D 或 3D 数组，实际 shape={data.shape}")

    n_bands = data.shape[0]

    if dtype is None:
        dtype = data.dtype

    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()
        profile.update(
            count=n_bands,
            dtype=dtype,
            compress=compress,
            driver="GTiff",
        )

        if nodata is not None:
            profile["nodata"] = nodata

        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(n_bands):
                dst.write(data[i].astype(dtype), i + 1)
                if band_names and i < len(band_names):
                    dst.set_band_description(i + 1, band_names[i])

    with open(output_path, "rb") as f:
        binary = f.read()

    return output_path, binary


def export_index_geotiff(
    index_arr: np.ndarray,
    reference_path: str,
    output_path: Optional[str] = None,
    index_name: str = "index",
) -> Tuple[str, bytes]:
    """导出指数栅格（NDVI/MNDWI/EVI 等）为单波段 GeoTIFF"""
    return export_raster_geotiff(
        data=index_arr,
        reference_path=reference_path,
        output_path=output_path,
        dtype="float32",
        nodata=DEFAULT_NODATA,
        band_names=[index_name.upper()],
    )


def export_mask_geotiff(
    mask: np.ndarray,
    reference_path: str,
    output_path: Optional[str] = None,
    mask_name: str = "mask",
) -> Tuple[str, bytes]:
    """导出二值/分类掩膜为单波段 GeoTIFF"""
    # 确保 mask 是整数类型
    if mask.dtype.kind == "f":
        mask = mask.astype(np.uint8)

    return export_raster_geotiff(
        data=mask,
        reference_path=reference_path,
        output_path=output_path,
        dtype="uint8",
        nodata=255,
        band_names=[mask_name],
    )


def export_classification_geotiff(
    class_arr: np.ndarray,
    reference_path: str,
    class_names: List[str],
    output_path: Optional[str] = None,
    colormap: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> Tuple[str, bytes]:
    """
    导出分类结果为彩色 GeoTIFF。

    参数:
        class_arr: 分类标签数组 (2D, uint8)
        reference_path: 参考影像
        class_names: 类别名称列表
        colormap: {class_id: (R,G,B)} 可选颜色映射
    """
    from rasterio.enums import ColorInterp

    if colormap is None:
        # 默认 6 类干旱区配色
        colormap = {
            0: (0, 0, 0),        # 背景 - 黑
            1: (30, 60, 180),    # 水体 - 蓝
            2: (35, 140, 35),    # 植被 - 绿
            3: (210, 180, 140),  # 裸地 - 棕
            4: (220, 50, 50),    # 建设用地 - 红
            5: (240, 220, 50),   # 农田 - 黄
            6: (128, 128, 128),  # 矿区 - 灰
        }

    import rasterio

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        output_path = tmp.name
        tmp.close()

    # 生成 RGB 三波段
    h, w = class_arr.shape
    rgb = np.zeros((3, h, w), dtype=np.uint8)
    for class_id, color in colormap.items():
        mask = class_arr == class_id
        for c in range(3):
            rgb[c][mask] = color[c]

    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()
        profile.update(
            count=3,
            dtype="uint8",
            compress=COMPRESSION,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(3):
                dst.write(rgb[i], i + 1)

    with open(output_path, "rb") as f:
        binary = f.read()

    return output_path, binary


# ============================================
# CSV 导出
# ============================================

def export_csv(
    data: Union[pd.DataFrame, Dict[str, Any], List[Dict]],
    output_path: Optional[str] = None,
    encoding: str = "utf-8-sig",
    index: bool = False,
) -> Tuple[str, bytes]:
    """
    导出数据为 CSV 文件。

    参数:
        data: DataFrame / dict / list of dict
        output_path: 输出路径
        encoding: 文件编码
        index: 是否包含行索引

    返回:
        (文件路径, CSV 二进制内容)
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data

    csv_content = df.to_csv(index=index, encoding=encoding)

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        output_path = tmp.name
        tmp.close()

    with open(output_path, "w", encoding=encoding) as f:
        f.write(csv_content)

    binary = csv_content.encode(encoding)
    return output_path, binary


def export_stats_csv(
    stats: Dict[str, Any],
    output_path: Optional[str] = None,
    label: str = "统计结果",
) -> Tuple[str, bytes]:
    """导出一行统计结果为 CSV"""
    row = {"指标": label}
    row.update(stats)
    return export_csv(pd.DataFrame([row]), output_path)


def export_timeseries_csv(
    dates: List[str],
    values: List[float],
    value_name: str = "value",
    output_path: Optional[str] = None,
) -> Tuple[str, bytes]:
    """导出时序数据为 CSV"""
    df = pd.DataFrame({"日期": dates, value_name: values})
    return export_csv(df, output_path)


# ============================================
# 图表导出 (PNG/SVG)
# ============================================

def export_figure(
    fig: plt.Figure,
    output_path: Optional[str] = None,
    fmt: str = "png",
    dpi: int = 150,
    bbox_inches: str = "tight",
) -> Tuple[str, bytes]:
    """
    导出 matplotlib 图为 PNG/SVG。

    返回:
        (文件路径, 图片二进制内容)
    """
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False)
        output_path = tmp.name
        tmp.close()

    fig.savefig(output_path, format=fmt, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)

    with open(output_path, "rb") as f:
        binary = f.read()

    return output_path, binary


def export_array_as_image(
    arr: np.ndarray,
    output_path: Optional[str] = None,
    cmap: str = "viridis",
    title: str = "",
    colorbar_label: str = "",
    fmt: str = "png",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Tuple[str, bytes]:
    """
    将 2D numpy 数组渲染为伪彩色图片。

    参数:
        arr: 2D 数组
        cmap: matplotlib colormap
        title: 图片标题
        colorbar_label: colorbar 标签
        fmt: png / svg
        vmin/vmax: 颜色范围
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar_label:
        plt.colorbar(im, ax=ax, label=colorbar_label, shrink=0.8)
    if title:
        ax.set_title(title, fontsize=14)
    ax.axis("off")

    return export_figure(fig, output_path, fmt=fmt)


def export_comparison_figure(
    arr1: np.ndarray,
    arr2: np.ndarray,
    diff: np.ndarray,
    label1: str = "t1",
    label2: str = "t2",
    cmap: str = "RdYlGn",
    output_path: Optional[str] = None,
    fmt: str = "png",
) -> Tuple[str, bytes]:
    """
    导出三面板对比图（用于变化检测等场景）。

    布局: [图1 | 图2 | 差值]
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(arr1, cmap=cmap)
    axes[0].set_title(label1, fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(arr2, cmap=cmap)
    axes[1].set_title(label2, fontsize=12)
    axes[1].axis("off")

    # 差值用发散色带
    im = axes[2].imshow(diff, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[2].set_title(f"{label1} - {label2} 差值", fontsize=12)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    return export_figure(fig, output_path, fmt=fmt)


def export_timeseries_plot(
    dates: List[str],
    values: List[float],
    value_name: str = "指数值",
    title: str = "时序变化",
    output_path: Optional[str] = None,
    fmt: str = "png",
    trend_line: bool = True,
) -> Tuple[str, bytes]:
    """导出一维时序折线图"""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(dates, values, "o-", linewidth=2, markersize=6, color="#1f77b4")
    ax.set_xlabel("日期", fontsize=11)
    ax.set_ylabel(value_name, fontsize=11)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    if trend_line and len(values) > 2:
        x = np.arange(len(values))
        z = np.polyfit(x, values, 1)
        trend = np.polyval(z, x)
        ax.plot(dates, trend, "--", color="red", linewidth=1.5, label="趋势线")
        ax.legend()

    plt.tight_layout()
    return export_figure(fig, output_path, fmt=fmt)


# ============================================
# GeoJSON 矢量导出
# ============================================

def raster_to_geojson_polygons(
    mask: np.ndarray,
    reference_path: str,
    output_path: Optional[str] = None,
    min_area_pixels: int = 10,
    simplify_tolerance: float = 0.5,
    class_values: Optional[Dict[int, str]] = None,
) -> Tuple[str, bytes]:
    """
    将栅格掩膜转为 GeoJSON 面矢量。

    参数:
        mask: 整数标签掩膜 (2D)
        reference_path: 参考 GeoTIFF（CRS + transform）
        output_path: 输出 .geojson 路径
        min_area_pixels: 最小面面积（像素），过小的多边形将被过滤
        simplify_tolerance: Douglas-Peucker 简化容差（像素单位）
        class_values: {class_id: "label"} 类别映射

    返回:
        (文件路径, GeoJSON 文本内容)
    """
    import rasterio
    from rasterio import features
    import geopandas as gpd
    from shapely.geometry import shape

    with rasterio.open(reference_path) as src:
        transform = src.transform
        crs = src.crs

    features_list = []
    unique_classes = np.unique(mask)

    for class_id in unique_classes:
        if class_id == 0 and class_values and 0 not in class_values:
            continue  # 跳过后台类

        class_mask = (mask == class_id).astype(np.uint8)

        # 矢量化
        shapes = features.shapes(class_mask, mask=class_mask, transform=transform)

        for geom, value in shapes:
            if value == 0:
                continue
            polygon = shape(geom)

            # 过滤小多边形
            if polygon.area < min_area_pixels:
                continue

            # 简化
            if simplify_tolerance > 0:
                polygon = polygon.simplify(simplify_tolerance, preserve_topology=True)

            if not polygon.is_empty:
                props = {"class_id": int(class_id)}
                if class_values and class_id in class_values:
                    props["class_name"] = class_values[class_id]
                features_list.append({
                    "type": "Feature",
                    "geometry": polygon.__geo_interface__,
                    "properties": props,
                })

    geojson_dict = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": str(crs)}},
        "features": features_list,
    }

    geojson_str = json.dumps(geojson_dict, ensure_ascii=False, indent=2)

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".geojson", delete=False)
        output_path = tmp.name
        tmp.close()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(geojson_str)

    return output_path, geojson_str.encode("utf-8")


def export_change_polygons_geojson(
    change_mask: np.ndarray,
    reference_path: str,
    output_path: Optional[str] = None,
    min_area_pixels: int = 15,
) -> Tuple[str, bytes]:
    """
    导出变化检测结果为 GeoJSON。

    change_mask 编码:
        0 = 无变化
        1 = 减少 (loss)
        2 = 增加 (gain)
    """
    return raster_to_geojson_polygons(
        mask=change_mask,
        reference_path=reference_path,
        output_path=output_path,
        min_area_pixels=min_area_pixels,
        class_values={1: "减少", 2: "增加"},
    )


# ============================================
# NumPy 中间结果导出
# ============================================

def export_numpy(
    arr: np.ndarray,
    output_path: Optional[str] = None,
    compress: bool = True,
) -> Tuple[str, bytes]:
    """导出 numpy 数组为 .npy 或 .npz 文件"""
    if output_path is None:
        suffix = ".npz" if compress else ".npy"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        output_path = tmp.name
        tmp.close()

    if compress:
        np.savez_compressed(output_path, data=arr)
    else:
        np.save(output_path, arr)

    with open(output_path, "rb") as f:
        binary = f.read()

    return output_path, binary


# ============================================
# Streamlit 辅助: 生成 download_button
# ============================================

def make_download_label(
    prefix: str,
    ext: str,
    timestamp: bool = True,
) -> str:
    """生成带时间戳的文件名"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp else ""
    if ts:
        return f"{prefix}_{ts}.{ext}"
    return f"{prefix}.{ext}"


# ============================================
# 批量导出
# ============================================

def export_batch(
    items: List[Dict[str, Any]],
    output_dir: str,
) -> List[str]:
    """
    批量导出多种格式结果。

    items 格式:
        [{"type": "geotiff", "data": arr, "reference": path, "name": "ndvi"},
         {"type": "csv", "data": df, "name": "stats"},
         {"type": "png", "data": fig, "name": "chart"},
         {"type": "geojson", "data": mask, "reference": path, "name": "water"},
         ...]

    返回:
        生成的文件路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    outputs = []

    for i, item in enumerate(items):
        item_type = item.get("type", "geotiff")
        name = item.get("name", f"export_{i:03d}")

        try:
            if item_type == "geotiff":
                path, _ = export_raster_geotiff(
                    data=item["data"],
                    reference_path=item["reference"],
                    output_path=os.path.join(output_dir, f"{name}.tif"),
                    dtype=item.get("dtype"),
                    nodata=item.get("nodata"),
                    band_names=item.get("band_names"),
                )
            elif item_type == "csv":
                path, _ = export_csv(
                    data=item["data"],
                    output_path=os.path.join(output_dir, f"{name}.csv"),
                )
            elif item_type == "png":
                path, _ = export_figure(
                    fig=item["data"],
                    output_path=os.path.join(output_dir, f"{name}.png"),
                )
            elif item_type == "geojson":
                path, _ = raster_to_geojson_polygons(
                    mask=item["data"],
                    reference_path=item["reference"],
                    output_path=os.path.join(output_dir, f"{name}.geojson"),
                    class_values=item.get("class_values"),
                )
            elif item_type == "npy":
                path, _ = export_numpy(
                    arr=item["data"],
                    output_path=os.path.join(output_dir, f"{name}.npy"),
                )
            else:
                raise ValueError(f"不支持的类型: {item_type}")

            outputs.append(path)
        except Exception as e:
            print(f"导出 {name} ({item_type}) 失败: {e}")

    return outputs
