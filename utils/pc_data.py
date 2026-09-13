"""
Planetary Computer 卫星数据获取模块
支持 Sentinel-2、Landsat 数据搜索、预览和下载
从 config.py 读取统一配置
"""

import pystac_client
import planetary_computer
import numpy as np
from PIL import Image
import io
import shutil
import os
import warnings
import requests
from urllib.parse import quote

from config import COLLECTIONS, CACHE_CONFIG, CACHE_DIR

warnings.filterwarnings("ignore")

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


def _geo_cache_dir() -> str:
    path = os.path.join(CACHE_DIR, "geotiffs")
    os.makedirs(path, exist_ok=True)
    return path


def _safe_cache_token(value: str) -> str:
    token = "".join(ch for ch in str(value) if ch.isalnum() or ch in "._-")
    return token[:140] or "item"


def _copy_cache_to_output(cache_path: str, output_path: str) -> bool:
    if not cache_path or cache_path == output_path:
        return False
    try:
        shutil.copyfile(cache_path, output_path)
        return True
    except OSError:
        return False


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
def _is_demo_mode():
    # 开关状态的唯一真源在 utils.demo_mode (UI 与底层模块共用)
    from utils.demo_mode import is_demo_mode
    return is_demo_mode()


def search_images(
    bbox,
    start_date,
    end_date,
    collection="Sentinel-2 L2A",
    cloud_cover_max=20,
    max_items=10,
):
    if _is_demo_mode():
        from utils.demo_data import build_demo_search_results
        return build_demo_search_results(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            collection=collection,
            cloud_cover_max=cloud_cover_max,
            max_items=max_items,
        )
    return _search_images_real_cached(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        collection=collection,
        cloud_cover_max=cloud_cover_max,
        max_items=max_items,
    )


@_cache(ttl=CACHE_CONFIG["ttl_medium"])
def _search_images_real_cached(
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

    stac_query = {"eo:cloud_cover": {"lt": cloud_cover_max}}
    platform_map = {
        "Landsat-8": ["landsat-8"],
        "Landsat-9": ["landsat-9"],
        "Landsat-7": ["landsat-7"],
        "Landsat-4-5": ["landsat-4", "landsat-5"],
    }
    if collection in platform_map:
        stac_query["platform"] = {"in": platform_map[collection]}

    search = catalog.search(
        collections=[collection_id],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query=stac_query,
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

        # 根据卫星类型选择波段名 (PC 渲染服务: Sentinel 用资产名, Landsat 用短名称)
        if "Sentinel" in collection:
            assets_str = "&assets=B04&assets=B03&assets=B02"
        else:  # Landsat (短名称: blue/green/red)
            assets_str = "&assets=red&assets=green&assets=blue"

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
        else:  # Landsat (短名称: nir08/red)
            expression = "(nir08-red)%2F(nir08%2Bred)"

        url = (
            f"https://planetarycomputer.microsoft.com/api/data/v1/item/preview.png"
            f"?collection={collection_id}"
            f"&item={quote(item_id)}"
            f"&expression={expression}"
            f"&asset_as_band=True"
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
        else:  # Landsat (短名称: green/swir16)
            expression = "(green-swir16)%2F(green%2Bswir16)"

        url = (
            f"https://planetarycomputer.microsoft.com/api/data/v1/item/preview.png"
            f"?collection={collection_id}"
            f"&item={quote(item_id)}"
            f"&expression={expression}"
            f"&asset_as_band=True"
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
# 下载任意 STAC 资产 (如 Landsat 热红外波段)
# ============================================
def download_asset(item, asset_name, output_path):
    """Download a named STAC asset as a single-band GeoTIFF."""
    cache_path = os.path.join(
        _geo_cache_dir(),
        f"{_safe_cache_token(item.id)}_{_safe_cache_token(asset_name)}.tif",
    )
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1024:
        _copy_cache_to_output(cache_path, output_path)
        return output_path

    try:
        import rioxarray
        href = item.assets[asset_name].href
        data = rioxarray.open_rasterio(href).squeeze()
        data.rio.to_raster(cache_path)
        _copy_cache_to_output(cache_path, output_path)
        return output_path
    except Exception as e:
        print(f"资产下载失败: {e}")
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
    cache_path = os.path.join(
        _geo_cache_dir(),
        f"{_safe_cache_token(item.id)}_{_safe_cache_token(collection)}_{_safe_cache_token(band_name)}.tif",
    )
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1024:
        _copy_cache_to_output(cache_path, output_path)
        return output_path

    try:
        import rioxarray

        bands = COLLECTIONS[collection]["bands"]
        band_key = bands[band_name]

        href = item.assets[band_key].href
        data = rioxarray.open_rasterio(href).squeeze()
        if data.ndim == 3:
            band_index = list(bands.keys()).index(band_name)
            data = data[band_index] if band_index < len(data) else data[0]
        data.rio.to_raster(cache_path)
        _copy_cache_to_output(cache_path, output_path)

        return output_path
    except Exception as e:
        print(f"下载失败: {e}")
        return None


# ============================================
# 批量下载多波段 (GeoTIFF)
# ============================================
def download_multiband(item, output_path, collection="Sentinel-2 L2A", band_names=None,
                       progress_callback=None):
    """
    下载多个波段并合成为一个多波段 GeoTIFF

    参数:
        item: STAC Item 对象 或 dict (含 id 键)
        output_path: 输出文件路径
        collection: 数据集名称
        band_names: 波段名称列表，默认 6 波段
        progress_callback: 进度回调 callback(completed, total) — 每完成一个波段调用,
                           用于大影像下载的进度展示

    返回:
        str: 成功返回输出路径，失败返回 None
    """
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
        if progress_callback:
            try:
                progress_callback(1, 1)
            except Exception:
                pass
        return output_path

    bands = COLLECTIONS[collection]["bands"]
    if band_names is None:
        band_names = ["blue", "green", "red", "nir", "swir1", "swir2"]

    band_token = "_".join(_safe_cache_token(b) for b in band_names)
    cache_path = os.path.join(
        _geo_cache_dir(),
        f"{_safe_cache_token(item.id)}_{_safe_cache_token(collection)}_{band_token}.tif",
    )

    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1024:
        _copy_cache_to_output(cache_path, output_path)
        if progress_callback:
            try:
                progress_callback(1, 1)
            except Exception:
                pass
        return output_path


    try:
        import rasterio
        from utils.logging_config import LogTimer

        band_keys = [bands[b] for b in band_names]

        # 读取第一个波段获取元数据
        first_href = item.assets[band_keys[0]].href
        import rioxarray

        first_data = rioxarray.open_rasterio(first_href).squeeze()

        # 读取所有波段
        all_bands = []
        total_bands = len(band_keys)
        # 6 波段下载是全平台最慢的单步操作 (30s-2min), 服务端需要留耗时记录
        with LogTimer(f"下载多波段 {item.id} ({total_bands}波段)"):
            for i, bk in enumerate(band_keys):
                href = item.assets[bk].href
                data = rioxarray.open_rasterio(href).squeeze()
                # 重采样到统一尺寸
                if data.shape != first_data.shape:
                    data = data.rio.reproject_match(first_data)
                band_values = data.values
                if band_values.ndim == 3:
                    band_values = band_values[i] if i < len(band_values) else band_values[0]
                all_bands.append(band_values.astype(first_data.dtype))
                # 进度回调 (每完成一个波段)
                if progress_callback:
                    try:
                        progress_callback(i + 1, total_bands)
                    except Exception:
                        pass

        # 写入多波段 TIFF
        stack = np.stack(all_bands, axis=0)
        with rasterio.open(
            cache_path,
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

        _copy_cache_to_output(cache_path, output_path)
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
    """
    if item_id.startswith("demo_"):
        from utils.demo_data import get_demo_rgb
        return get_demo_rgb(item_id, collection, width=width)
    catalog = get_catalog()
    collection_id = COLLECTIONS[collection]["id"]
    item = catalog.get_collection(collection_id).get_item(item_id)
    return get_rgb_preview(item, collection=collection, width=width)


@_cache(ttl=CACHE_CONFIG["ttl_medium"])
def get_ndvi_preview_cached(item_id: str, collection: str = "Sentinel-2 L2A", width: int = 512):
    """缓存版 NDVI 预览 — 基于 item_id"""
    if item_id.startswith("demo_"):
        from utils.demo_data import get_demo_rgb
        return get_demo_rgb(item_id, collection, width=width)
    catalog = get_catalog()
    collection_id = COLLECTIONS[collection]["id"]
    item = catalog.get_collection(collection_id).get_item(item_id)
    return get_ndvi_preview(item, collection=collection, width=width)


@_cache(ttl=CACHE_CONFIG["ttl_medium"])
def get_mndwi_preview_cached(item_id: str, collection: str = "Sentinel-2 L2A", width: int = 512):
    """缓存版 MNDWI 预览 — 基于 item_id"""
    if item_id.startswith("demo_"):
        from utils.demo_data import get_demo_rgb
        return get_demo_rgb(item_id, collection, width=width)
    catalog = get_catalog()
    collection_id = COLLECTIONS[collection]["id"]
    item = catalog.get_collection(collection_id).get_item(item_id)
    return get_mndwi_preview(item, collection=collection, width=width)
