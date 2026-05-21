"""
Planetary Computer 卫星数据获取模块
支持 Sentinel-2、Landsat 数据搜索、预览和下载
从 config.py 读取统一配置
"""

import pystac_client
import planetary_computer
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import warnings
import requests
from urllib.parse import quote

warnings.filterwarnings("ignore")

# 从统一配置导入
from config import COLLECTIONS, STUDY_AREAS, CACHE_CONFIG

# ---- 条件缓存装饰器 ----
try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False


def _cache(ttl: int):
    """条件缓存装饰器: Streamlit 环境启用缓存，否则退化为 no-op"""
    if _HAS_STREAMLIT:
        return st.cache_data(ttl=ttl, show_spinner=CACHE_CONFIG.get("show_spinner", True))
    else:
        def noop_decorator(func):
            return func
        return noop_decorator


# ============================================
# STAC 目录连接
# ============================================
def get_catalog():
    """连接 Planetary Computer STAC 目录"""
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    return catalog


# ============================================
# 搜索影像
# ============================================
@_cache(ttl=CACHE_CONFIG["ttl_medium"])
def search_images(
    bbox,
    start_date,
    end_date,
    collection="Sentinel-2 L2A",
    cloud_cover_max=20,
    max_items=10,
):
    """
    搜索卫星影像

    参数:
        bbox: [min_lon, min_lat, max_lon, max_lat] 边界框
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        collection: 数据集名称 (key in COLLECTIONS)
        cloud_cover_max: 最大云量 (%)
        max_items: 最大返回数量

    返回:
        list: 影像列表，每项包含 datetime, cloud_cover, id 等信息
    """
    """
    搜索卫星影像

    参数:
        bbox: [min_lon, min_lat, max_lon, max_lat] 边界框
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        collection: 数据集名称 (key in COLLECTIONS)
        cloud_cover_max: 最大云量 (%)
        max_items: 最大返回数量

    返回:
        list: 影像列表，每项包含 datetime, cloud_cover, id 等信息
    """
    catalog = get_catalog()

    collection_id = COLLECTIONS[collection]["id"]

    search = catalog.search(
        collections=[collection_id],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": cloud_cover_max}},
        max_items=max_items,
        sortby=[{"field": "datetime", "direction": "desc"}],
    )

    items = list(search.items())

    results = []
    for item in items:
        # 获取缩略图URL
        thumbnail_url = None
        if "thumbnail" in item.assets:
            thumbnail_url = item.assets["thumbnail"].href

        results.append(
            {
                "id": item.id,
                "datetime": item.datetime.strftime("%Y-%m-%d"),
                "cloud_cover": item.properties.get("eo:cloud_cover", "N/A"),
                "bbox": item.bbox,
                "thumbnail_url": thumbnail_url,
                "item": item,
            }
        )

    return results


# ============================================
# 获取缩略图 (快速)
# ============================================
def get_thumbnail(item, timeout=15):
    """
    获取影像缩略图 (最快方式)

    参数:
        item: STAC Item 对象
        timeout: 超时时间(秒)

    返回:
        PIL.Image 或 None
    """
    try:
        if "thumbnail" in item.assets:
            url = item.assets["thumbnail"].href
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content))
    except Exception as e:
        print(f"缩略图加载失败: {e}")

    return None


# ============================================
# 获取 RGB 预览 (使用 Planetary Computer 渲染服务)
# ============================================
def get_rgb_preview(item, collection="Sentinel-2 L2A", width=512):
    """
    使用 Planetary Computer 渲染服务获取 RGB 预览

    参数:
        item: STAC Item 对象
        collection: 数据集名称 (key in COLLECTIONS)
        width: 图像宽度

    返回:
        PIL.Image 或 None
    """
    try:
        collection_id = COLLECTIONS[collection]["id"]
        item_id = item.id

        # 根据卫星类型选择波段名
        if "Sentinel" in collection:
            assets_str = "&assets=B04&assets=B03&assets=B02"
        else:  # Landsat
            assets_str = "&assets=SR_B4&assets=SR_B3&assets=SR_B2"

        url = (
            f"https://planetarycomputer.microsoft.com/api/data/v1/item/preview.png"
            f"?collection={collection_id}"
            f"&item={quote(item_id)}"
            f"{assets_str}"
            f"&rescale=0,2000"
            f"&width={width}"
        )

        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))

        # 回退到缩略图
        return get_thumbnail(item)

    except Exception as e:
        print(f"RGB预览失败: {e}")
        return get_thumbnail(item)


# ============================================
# 获取 NDVI 预览 (使用渲染服务)
# ============================================
def get_ndvi_preview(item, collection="Sentinel-2 L2A", width=512):
    """使用 Planetary Computer 渲染服务获取 NDVI 预览"""
    try:
        collection_id = COLLECTIONS[collection]["id"]
        item_id = item.id

        if "Sentinel" in collection:
            expression = "(B08-B04)%2F(B08%2BB04)"
        else:  # Landsat
            expression = "(SR_B5-SR_B4)%2F(SR_B5%2BSR_B4)"

        url = (
            f"https://planetarycomputer.microsoft.com/api/data/v1/item/preview.png"
            f"?collection={collection_id}"
            f"&item={quote(item_id)}"
            f"&expression={expression}"
            f"&rescale=-1,1"
            f"&colormap_name=rdylgn"
            f"&width={width}"
        )

        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))

    except Exception as e:
        print(f"NDVI预览失败: {e}")

    return None


# ============================================
# 获取 MNDWI 预览 (使用渲染服务)
# ============================================
def get_mndwi_preview(item, collection="Sentinel-2 L2A", width=512):
    """使用 Planetary Computer 渲染服务获取 MNDWI 预览"""
    try:
        collection_id = COLLECTIONS[collection]["id"]
        item_id = item.id

        if "Sentinel" in collection:
            expression = "(B03-B11)%2F(B03%2BB11)"
        else:  # Landsat
            expression = "(SR_B3-SR_B6)%2F(SR_B3%2BSR_B6)"

        url = (
            f"https://planetarycomputer.microsoft.com/api/data/v1/item/preview.png"
            f"?collection={collection_id}"
            f"&item={quote(item_id)}"
            f"&expression={expression}"
            f"&rescale=-1,1"
            f"&colormap_name=blues"
            f"&width={width}"
        )

        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))

    except Exception as e:
        print(f"MNDWI预览失败: {e}")

    return None


# ============================================
# 下载完整波段 (GeoTIFF)
# ============================================
def download_band(item, band_name, output_path, collection="Sentinel-2 L2A"):
    """
    下载单个波段为 GeoTIFF

    参数:
        item: STAC Item 对象
        band_name: 波段名称 (blue, green, red, nir, swir1, swir2)
        output_path: 输出文件路径
        collection: 数据集名称 (key in COLLECTIONS)

    返回:
        str: 成功返回输出路径，失败返回 None
    """
    try:
        import rioxarray

        bands = COLLECTIONS[collection]["bands"]
        band_key = bands[band_name]

        href = item.assets[band_key].href
        data = rioxarray.open_rasterio(href).squeeze()
        data.rio.to_raster(output_path)

        return output_path
    except Exception as e:
        print(f"下载失败: {e}")
        return None


# ============================================
# 批量下载多波段 (GeoTIFF)
# ============================================
def download_multiband(item, output_path, collection="Sentinel-2 L2A", band_names=None):
    """
    下载多个波段并合成为一个多波段 GeoTIFF

    参数:
        item: STAC Item 对象 或 dict (含 id 键)
        output_path: 输出文件路径
        collection: 数据集名称
        band_names: 波段名称列表，默认 6 波段

    返回:
        str: 成功返回输出路径，失败返回 None
    """
    # 缓存命中: 文件已存在且有效
    import os
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
        return output_path

    try:
        import rasterio
        from rasterio.transform import from_bounds

        bands = COLLECTIONS[collection]["bands"]
        if band_names is None:
            band_names = ["blue", "green", "red", "nir", "swir1", "swir2"]

        band_keys = [bands[b] for b in band_names]

        # 读取第一个波段获取元数据
        first_href = item.assets[band_keys[0]].href
        import rioxarray

        first_data = rioxarray.open_rasterio(first_href).squeeze()

        # 读取所有波段
        all_bands = []
        for bk in band_keys:
            href = item.assets[bk].href
            data = rioxarray.open_rasterio(href).squeeze()
            # 重采样到统一尺寸
            if data.shape != first_data.shape:
                data = data.rio.reproject_match(first_data)
            all_bands.append(data.values.astype(first_data.dtype))

        # 写入多波段 TIFF
        stack = np.stack(all_bands, axis=0)
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=stack.shape[1],
            width=stack.shape[2],
            count=stack.shape[0],
            dtype=stack.dtype,
            crs=first_data.rio.crs,
            transform=first_data.rio.transform(),
        ) as dst:
            dst.write(stack)

        return output_path
    except Exception as e:
        print(f"多波段下载失败: {e}")
        return None


# ============================================
# 带缓存的预览包装 (基于 item_id，绕过 STAC Item 不可哈希的限制)
# ============================================

@_cache(ttl=CACHE_CONFIG["ttl_medium"])
def get_rgb_preview_cached(item_id: str, collection: str = "Sentinel-2 L2A", width: int = 512):
    """
    缓存版 RGB 预览 — 通过 Planetary Computer 渲染服务获取。
    使用 item_id 而非 Item 对象作为缓存键，绕过 STAC Item 不可哈希问题。

    注意: 此函数需要一次 STAC API 调用来获取 Item 对象。
    推荐在页面中传递已有的 Item json 数据。
    """
    catalog = get_catalog()
    collection_id = COLLECTIONS[collection]["id"]
    item = catalog.get_item(item_id, collection_id)
    return get_rgb_preview(item, collection=collection, width=width)


@_cache(ttl=CACHE_CONFIG["ttl_medium"])
def get_ndvi_preview_cached(item_id: str, collection: str = "Sentinel-2 L2A", width: int = 512):
    """缓存版 NDVI 预览 — 基于 item_id"""
    catalog = get_catalog()
    collection_id = COLLECTIONS[collection]["id"]
    item = catalog.get_item(item_id, collection_id)
    return get_ndvi_preview(item, collection=collection, width=width)


@_cache(ttl=CACHE_CONFIG["ttl_medium"])
def get_mndwi_preview_cached(item_id: str, collection: str = "Sentinel-2 L2A", width: int = 512):
    """缓存版 MNDWI 预览 — 基于 item_id"""
    catalog = get_catalog()
    collection_id = COLLECTIONS[collection]["id"]
    item = catalog.get_item(item_id, collection_id)
    return get_mndwi_preview(item, collection=collection, width=width)
