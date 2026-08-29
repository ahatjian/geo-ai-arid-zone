"""
土地覆盖数据接入模块
基于 ESA WorldCover (10m, 11类) 和 ESRI Land Cover (10m, 9类)
通过 Cloud-Optimized GeoTIFF (COG) 远程读取，无需全量下载

依赖: rasterio, numpy
"""

import os
import math
import tempfile
import warnings
import numpy as np
from typing import Optional, List, Dict, Tuple

warnings.filterwarnings("ignore")

# 从统一配置导入缓存 TTL
from config import CACHE_CONFIG

# ---- 条件缓存装饰器 ----
try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False


def _cache(ttl: int):
    """条件缓存装饰器"""
    if _HAS_STREAMLIT:
        return st.cache_data(ttl=ttl, show_spinner=CACHE_CONFIG.get("show_spinner", True))
    else:
        def noop_decorator(func):
            return func
        return noop_decorator

# ============================================
# ESA WorldCover 配置
# ============================================

ESA_S3_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
ESA_VERSION = "v200"
ESA_YEAR = "2021"

# ESA WorldCover 11 类定义
ESA_CLASSES = {
    0:  {"name": "无数据",           "color": "#000000", "code": "nodata"},
    10: {"name": "林地 (Tree cover)", "color": "#006400", "code": "tree_cover"},
    20: {"name": "灌木 (Shrubland)",  "color": "#FFBB22", "code": "shrubland"},
    30: {"name": "草地 (Grassland)",  "color": "#FFFF4C", "code": "grassland"},
    40: {"name": "农田 (Cropland)",   "color": "#F096FF", "code": "cropland"},
    50: {"name": "建设用地 (Built-up)", "color": "#FA0000", "code": "built_up"},
    60: {"name": "裸地/稀疏植被 (Barren)", "color": "#B4B4B4", "code": "barren"},
    70: {"name": "冰雪 (Snow/Ice)",     "color": "#F0F0F0", "code": "snow_ice"},
    80: {"name": "水域 (Open water)",   "color": "#0064C8", "code": "open_water"},
    90: {"name": "草本湿地 (Herbaceous wetland)", "color": "#0096A0", "code": "herb_wetland"},
    95: {"name": "红树林 (Mangroves)",            "color": "#00CF75", "code": "mangroves"},
    100:{"name": "苔藓/地衣 (Moss/Lichen)",       "color": "#FAE6A0", "code": "moss_lichen"},
}

# ESA → 干旱区6类 归并映射
ESA_TO_ARID6 = {
    0:   0,  # 无数据 → 0
    10:  1,  # 林地 → 植被
    20:  1,  # 灌木 → 植被
    30:  1,  # 草地 → 植被
    40:  4,  # 农田 → 农田
    50:  3,  # 建设用地 → 建设用地
    60:  2,  # 裸地 → 裸地
    70:  0,  # 冰雪 → 其他（干旱区基本没有）
    80:  5,  # 水域 → 水体
    90:  1,  # 湿地 → 植被
    95:  1,  # 红树林 → 植被（干旱区基本没有）
    100: 1,  # 苔藓 → 植被
}

# 干旱区6类（与 config.py AI_MODELS["land_cover"] 一致）
ARID6_CLASSES = {
    0: {"name": "其他/无数据", "color": "#000000"},
    1: {"name": "植被",         "color": "#00CC44"},
    2: {"name": "裸地",         "color": "#CCCCCC"},
    3: {"name": "建设用地",      "color": "#FF4444"},
    4: {"name": "农田",         "color": "#FFAA00"},
    5: {"name": "水体",         "color": "#0066FF"},
}

# ============================================
# ESRI Land Cover 配置
# ============================================

ESRI_CLASSES = {
    1:  {"name": "水体 (Water)",              "color": "#1A5BAB", "code": "water"},
    2:  {"name": "林地 (Trees)",              "color": "#358221", "code": "trees"},
    4:  {"name": "淹水植被 (Flooded Vegetation)", "color": "#87D19E", "code": "flooded_veg"},
    5:  {"name": "农田 (Crops)",               "color": "#FFDB5C", "code": "crops"},
    7:  {"name": "建设用地 (Built Area)",        "color": "#ED022A", "code": "built_area"},
    8:  {"name": "裸地 (Bare Ground)",          "color": "#EDE9E4", "code": "bare_ground"},
    9:  {"name": "冰雪 (Snow/Ice)",             "color": "#F2FAFF", "code": "snow_ice"},
    10: {"name": "云 (Clouds)",                "color": "#C8C8C8", "code": "clouds"},
    11: {"name": "草地/牧场 (Rangeland)",         "color": "#C0AD8C", "code": "rangeland"},
}

ESRI_TO_ARID6 = {
    1:  5,  # 水体
    2:  1,  # 林地 → 植被
    4:  1,  # 淹水植被 → 植被
    5:  4,  # 农田
    7:  3,  # 建设用地
    8:  2,  # 裸地
    9:  0,  # 冰雪 → 其他
    10: 0,  # 云 → 其他
    11: 1,  # 草地 → 植被
}

# ESRI Living Atlas ImageServer URL (2024)
ESRI_LULC_SERVER = (
    "https://lulc.servicesdevus2.arcgis.com/Yo6tNx2f6DaLCGmt/arcgis/rest/services/"
    "Sentinel2_10m_LandCover/ImageServer"
)


# ============================================
# ESA 切片计算
# ============================================

def _esa_tile_name(lat_south: float, lon_west: float) -> str:
    """根据西南角坐标生成 ESA 切片名，如 N36E075"""
    lat_floor = int(math.floor(lat_south / 3) * 3)
    lon_floor = int(math.floor(lon_west / 3) * 3)
    lat_hemi = "N" if lat_floor >= 0 else "S"
    lon_hemi = "E" if lon_floor >= 0 else "W"
    return f"{lat_hemi}{abs(lat_floor):02d}{lon_hemi}{abs(lon_floor):03d}"


def _esa_tiles_for_bbox(bbox: List[float]) -> List[str]:
    """
    计算覆盖给定 bbox 的所有 ESA 3°×3° 切片

    参数:
        bbox: [min_lon, min_lat, max_lon, max_lat]

    返回:
        切片名列表
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    tiles = set()

    # 3° 网格
    lat_start = int(math.floor(min_lat / 3) * 3)
    lon_start = int(math.floor(min_lon / 3) * 3)

    lat = lat_start
    while lat < max_lat:
        lon = lon_start
        while lon < max_lon:
            tiles.add(_esa_tile_name(lat, lon))
            lon += 3
        lat += 3

    return sorted(tiles)


def _esa_tile_url(tile_name: str, version: str = ESA_VERSION,
                  year: str = ESA_YEAR) -> str:
    """构建 ESA WorldCover 切片 COG URL"""
    return (
        f"{ESA_S3_BASE}/{version}/{year}/map/"
        f"ESA_WorldCover_10m_{year}_{version}_{tile_name}_Map.tif"
    )


# ============================================
# COG 远程读取
# ============================================

def read_landcover_cog(
    url: str,
    bbox: Optional[List[float]] = None,
    target_crs: str = "EPSG:4326",
) -> Tuple[np.ndarray, dict]:
    """
    通过 HTTP 远程读取 COG 土地覆盖数据，只拉取 bbox 需要的窗口

    参数:
        url: COG 的 HTTPS URL
        bbox: [min_lon, min_lat, max_lon, max_lat]，None=读取全部
        target_crs: 目标坐标系

    返回:
        (class_array, meta) — class_array 是 uint8 分类数组，meta 是元数据字典
    """
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds

    with rasterio.open(url) as src:
        if bbox is not None:
            # 将 bbox 转换到数据本身的 CRS
            warped_bbox = transform_bounds(target_crs, src.crs, *bbox)
            window = from_bounds(*warped_bbox, transform=src.transform)
            data = src.read(1, window=window).astype(np.uint8)
            # 获取窗口对应的 transform
            win_transform = src.window_transform(window)
            meta = {
                "crs": str(src.crs),
                "transform": win_transform,
                "width": data.shape[1],
                "height": data.shape[0],
                "pixel_size": abs(win_transform.a),
                "nodata": src.nodata,
            }
        else:
            data = src.read(1).astype(np.uint8)
            meta = {
                "crs": str(src.crs),
                "transform": src.transform,
                "width": src.width,
                "height": src.height,
                "pixel_size": abs(src.transform.a),
                "nodata": src.nodata,
            }

    return data, meta


@_cache(ttl=CACHE_CONFIG["ttl_long"])
def get_esa_landcover(
    bbox: List[float],
    version: str = ESA_VERSION,
    year: str = ESA_YEAR,
) -> Tuple[np.ndarray, dict]:
    """
    获取 ESA WorldCover 土地覆盖分类数据

    对覆盖 bbox 的 ESA 切片逐一远程读取窗口，合并为单一大数组

    参数:
        bbox: [min_lon, min_lat, max_lon, max_lat]
        version: "v100" (2020) 或 "v200" (2021)
        year: "2020" 或 "2021"

    返回:
        (class_array, meta) — 像素值为 ESA 类别值 (10,20,...,100)
    """
    tiles = _esa_tiles_for_bbox(bbox)

    if not tiles:
        raise ValueError(f"bbox {bbox} 未覆盖任何 ESA 切片")

    if len(tiles) == 1:
        url = _esa_tile_url(tiles[0], version, year)
        return read_landcover_cog(url, bbox)

    # 多切片拼接
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds

    pieces = []
    for tile_name in tiles:
        url = _esa_tile_url(tile_name, version, year)
        try:
            data, meta_piece = read_landcover_cog(url, bbox)
            if data.size > 0:
                pieces.append((data, meta_piece))
        except Exception:
            continue  # 切片可能不覆盖该区域

    if len(pieces) == 1:
        return pieces[0]

    if not pieces:
        raise RuntimeError(f"无法从任何 ESA 切片读取数据: {tiles}")

    # 简单拼接策略：按 bbox 排序然后全景
    # 这里使用 rasterio merge 逻辑
    from rasterio.merge import merge
    from rasterio.io import MemoryFile

    memfiles = []
    for data, meta in pieces:
        with MemoryFile() as mem:
            with mem.open(
                driver="GTiff",
                width=meta["width"],
                height=meta["height"],
                count=1,
                dtype="uint8",
                crs=meta["crs"],
                transform=meta["transform"],
            ) as dst:
                dst.write(data, 1)
            memfiles.append(mem.open())

    merged, merged_transform = merge(memfiles, bounds=tuple(bbox),
                                      res=10, method="nearest")

    merged_data = merged[0].astype(np.uint8)
    merged_meta = {
        "crs": "EPSG:4326",
        "transform": merged_transform,
        "width": merged_data.shape[1],
        "height": merged_data.shape[0],
        "pixel_size": abs(merged_transform.a),
        "nodata": 0,
    }

    for f in memfiles:
        f.close()

    return merged_data, merged_meta


# ============================================
# 类别映射
# ============================================

def esa_to_arid6(esa_class: np.ndarray) -> np.ndarray:
    """将 ESA 11类数组映射到干旱区6类"""
    result = np.zeros_like(esa_class, dtype=np.uint8)
    for esa_val, arid_val in ESA_TO_ARID6.items():
        result[esa_class == esa_val] = arid_val
    return result


def esri_to_arid6(esri_class: np.ndarray) -> np.ndarray:
    """将 ESRI 9类数组映射到干旱区6类"""
    result = np.zeros_like(esri_class, dtype=np.uint8)
    for esri_val, arid_val in ESRI_TO_ARID6.items():
        result[esri_class == esri_val] = arid_val
    return result


# ============================================
# 统计
# ============================================

def compute_landcover_stats(
    class_array: np.ndarray,
    class_names: Dict[int, str],
    class_colors: Dict[int, str],
    pixel_size_m: float = 10.0,
) -> List[Dict]:
    """
    计算土地覆盖分类统计

    参数:
        class_array: 分类数组 (整数类别值)
        class_names: {class_val: "类别名"}
        class_colors: {class_val: "#RRGGBB"}
        pixel_size_m: 像元大小 (米)

    返回:
        list[dict]: 每个类别的统计信息
    """
    total_pixels = int(class_array.size)
    stats = []
    unique_vals = np.unique(class_array)

    for val in unique_vals:
        count = int(np.sum(class_array == val))
        ratio = count / total_pixels if total_pixels > 0 else 0
        area_km2 = count * (pixel_size_m ** 2) / 1e6

        stats.append({
            "class_value": int(val),
            "class_name": class_names.get(int(val), f"类别{int(val)}"),
            "class_color": class_colors.get(int(val), "#888888"),
            "pixel_count": count,
            "pixel_ratio": round(ratio, 4),
            "area_km2": round(area_km2, 4),
        })

    stats.sort(key=lambda x: -x["pixel_count"])  # 按像元数降序
    return stats


# ============================================
# 导出
# ============================================

def export_landcover_geotiff(
    class_array: np.ndarray,
    output_path: str,
    transform: any,
    crs: str = "EPSG:4326",
    nodata: int = 0,
    class_names: Optional[Dict[int, str]] = None,
) -> str:
    """
    将土地覆盖分类数组导出为 GeoTIFF

    参数:
        class_array: 分类数组
        output_path: 输出路径
        transform: affine transform
        crs: 坐标系
        nodata: 无数据值
        class_names: 类别名（写入元数据）

    返回:
        str: 输出文件路径
    """
    import rasterio

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=class_array.shape[0],
        width=class_array.shape[1],
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="DEFLATE",
    ) as dst:
        dst.write(class_array, 1)
        if class_names:
            # 写入类别说明到 TIFF 标签
            desc = "; ".join(f"{k}:{v}" for k, v in sorted(class_names.items()))
            dst.update_tags(CLASSIFICATION=desc)

    return output_path


# ============================================
# 一站式: 获取 + 映射 + 统计
# ============================================

@_cache(ttl=CACHE_CONFIG["ttl_long"])
def get_landcover_for_study_area(
    bbox: List[float],
    source: str = "esa",
    map_to_arid6: bool = True,
) -> dict:
    """
    一站式: 获取研究区的土地覆盖分类数据

    参数:
        bbox: [min_lon, min_lat, max_lon, max_lat]
        source: "esa" (WorldCover) 或 "esri" (ESRI Land Cover)
        map_to_arid6: 是否映射到干旱区6类

    返回:
        dict: {
            "class_array": np.ndarray,
            "meta": dict,
            "stats": list[dict],
            "class_names": dict,
            "class_colors": dict,
            "source": str,
            "success": bool,
            "error": str or None,
        }
    """
    result = {
        "class_array": None,
        "meta": {},
        "stats": [],
        "class_names": {},
        "class_colors": {},
        "source": source,
        "success": False,
        "error": None,
    }

    try:
        if source == "esa":
            raw_array, meta = get_esa_landcover(bbox)
            class_names = {v["code"]: v["name"] for v in ESA_CLASSES.values()
                          if v["code"] != "nodata"} if not map_to_arid6 else {}
            class_colors = {v["code"]: v["color"] for v in ESA_CLASSES.values()
                           if v["code"] != "nodata"} if not map_to_arid6 else {}
            result["source_label"] = "ESA WorldCover (2021 v200)"
        else:
            # ESRI: 通过 TileMap 服务
            raw_array, meta = _get_esri_landcover(bbox)
            class_names = {v["code"]: v["name"] for v in ESRI_CLASSES.values()} if not map_to_arid6 else {}
            class_colors = {v["code"]: v["color"] for v in ESRI_CLASSES.values()} if not map_to_arid6 else {}
            result["source_label"] = "ESRI Land Cover (2024 Sentinel-2)"

        if map_to_arid6:
            if source == "esa":
                class_array = esa_to_arid6(raw_array)
            else:
                class_array = esri_to_arid6(raw_array)
            class_names = {k: v["name"] for k, v in ARID6_CLASSES.items()}
            class_colors = {k: v["color"] for k, v in ARID6_CLASSES.items()}
        else:
            class_array = raw_array
            # 从对应的产品 class 映射中提取
            if source == "esa":
                class_names = {k: v["name"] for k, v in ESA_CLASSES.items() if k != 0}
                class_colors = {k: v["color"] for k, v in ESA_CLASSES.items() if k != 0}
            else:
                class_names = {k: v["name"] for k, v in ESRI_CLASSES.items()}
                class_colors = {k: v["color"] for k, v in ESRI_CLASSES.items()}

        # 修正像素大小：ESA COG 是 EPSG:4326，pixel_size 为度，需转为米
        pixel_size_deg = meta.get("pixel_size", 0.0000898)
        if source == "esa":
            # ESA WorldCover 原生 10m 分辨率
            pixel_size_m = 10.0
        elif source == "esri":
            # ESRI 通过 ImageServer 导出，计算度数到米的近似转换
            center_lat = (bbox[1] + bbox[3]) / 2
            meters_per_deg_lat = 111320.0
            meters_per_deg_lon = 111320.0 * math.cos(math.radians(center_lat))
            # 假设像素接近正方形，取 lat 方向
            pixel_size_m = pixel_size_deg * meters_per_deg_lat
        else:
            pixel_size_m = 10.0

        stats = compute_landcover_stats(class_array, class_names, class_colors, pixel_size_m)

        result["class_array"] = class_array
        result["meta"] = meta
        result["stats"] = stats
        result["class_names"] = class_names
        result["class_colors"] = class_colors
        result["success"] = True

    except ImportError as e:
        result["error"] = f"缺少依赖: {e}"
    except Exception as e:
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()

    return result


# ============================================
# ESRI Land Cover 辅助 (简化版)
# ============================================

def _get_esri_landcover(bbox: List[float]) -> Tuple[np.ndarray, dict]:
    """
    通过 ESRI Living Atlas ImageServer WCS 获取土地覆盖数据
    注：需要网络连接，可能较慢
    """
    # ESRI Land Cover 通过 ImageServer 的 exportImage 端点
    # 这里提供一个基于 URL 参数的实现
    import rasterio
    import io
    import requests

    min_lon, min_lat, max_lon, max_lat = bbox

    # 计算图像尺寸（10m分辨率）
    # 在纬度方向：1°≈111km
    width_px = int((max_lon - min_lon) * 111000 / 10)
    height_px = int((max_lat - min_lat) * 111000 / 10)

    # 限制大小
    max_dim = 2000
    if width_px > max_dim:
        height_px = int(height_px * max_dim / width_px)
        width_px = max_dim
    if height_px > max_dim:
        width_px = int(width_px * max_dim / height_px)
        height_px = max_dim

    url = (
        f"{ESRI_LULC_SERVER}/exportImage"
        f"?bbox={min_lon},{min_lat},{max_lon},{max_lat}"
        f"&bboxSR=4326&imageSR=4326"
        f"&size={width_px},{height_px}"
        f"&format=tiff&pixelType=U8&noData=0"
        f"&f=image"
    )

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with rasterio.open(io.BytesIO(response.content)) as src:
        data = src.read(1).astype(np.uint8)
        meta = {
            "crs": str(src.crs),
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "pixel_size": abs(src.transform.a),
            "nodata": 0,
        }

    return data, meta


# ============================================
# ESA 切片列表（预计算研究区）
# ============================================

def get_tile_list_for_area(area_name: str, study_areas: dict) -> List[Dict]:
    """
    获取研究区对应的 ESA 切片列表及下载链接 (兼容旧接口, 按研究区名)

    参数:
        area_name: 研究区名称
        study_areas: STUDY_AREAS 字典

    返回:
        list[dict]: 每个切片的名称、URL、边界
    """
    area = study_areas.get(area_name)
    if not area:
        return []
    return get_tile_list_for_bbox(area["bbox"])


def get_tile_list_for_bbox(bbox: List[float]) -> List[Dict]:
    """
    获取覆盖 bbox 的 ESA 切片列表及下载链接 (支持自定义 AOI)

    参数:
        bbox: [min_lon, min_lat, max_lon, max_lat]

    返回:
        list[dict]: 每个切片的名称、URL、边界
    """
    import re

    tiles = _esa_tiles_for_bbox(bbox)

    tile_info = []
    for tile_name in tiles:
        url = _esa_tile_url(tile_name)
        # 计算切片边界
        match = re.match(r"([NS])(\d{2})([EW])(\d{3})", tile_name)
        if match:
            hem_lat, deg_lat, hem_lon, deg_lon = match.groups()
            lat_sw = int(deg_lat) * (1 if hem_lat == "N" else -1)
            lon_sw = int(deg_lon) * (1 if hem_lon == "E" else -1)
            tile_bbox = [lon_sw, lat_sw, lon_sw + 3, lat_sw + 3]
        else:
            tile_bbox = bbox

        tile_info.append({
            "name": tile_name,
            "url": url,
            "bbox": tile_bbox,
            "size_hint": "~10 MB (COG, 可远程部分读取)",
        })

    return tile_info
