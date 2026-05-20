"""
Planetary Computer 卫星数据自动下载脚本 (推荐！无需额外注册)
直接从 Microsoft Planetary Computer 下载 Sentinel-2 / Landsat 数据
优点: 无需注册账号，国内可访问，数据格式为 COG (Cloud Optimized GeoTIFF)

使用方法:
1. 在 config.py 中设置参数
2. 运行: python download_planetary.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

# 自动安装所需库
required_packages = ['pystac-client', 'planetary-computer', 'rioxarray', 'xarray', 'tqdm', 'requests']
for package in required_packages:
    try:
        __import__(package.replace('-', '_'))
    except ImportError:
        print(f"正在安装 {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple'])

import pystac_client
import planetary_computer
import rioxarray
import numpy as np
from tqdm import tqdm
from PIL import Image

# 导入配置
try:
    from config import *
except ImportError:
    print("错误: 找不到 config.py 文件，请确保它在同一目录下")
    sys.exit(1)


def ensure_dir(path):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_catalog():
    """连接 Planetary Computer STAC 目录"""
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )
    return catalog


def search_data(catalog, collection_id, bbox, start_date, end_date, max_cloud=20, max_items=50):
    """搜索影像"""
    search = catalog.search(
        collections=[collection_id],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud}} if max_cloud < 100 else None,
        max_items=max_items,
        sortby=[{"field": "datetime", "direction": "desc"}]
    )
    
    items = list(search.items())
    
    results = []
    for item in items:
        results.append({
            "id": item.id,
            "datetime": item.datetime.strftime("%Y-%m-%d"),
            "cloud_cover": item.properties.get("eo:cloud_cover", "N/A"),
            "item": item
        })
    
    return results


def download_band(item, band_key, output_path, collection="Sentinel-2"):
    """
    下载单个波段
    
    参数:
        item: STAC Item
        band_key: 波段名称 (如 'B04', 'B03', 'B02')
        output_path: 输出路径
        collection: 数据集
    
    返回:
        bool: 是否成功
    """
    try:
        if band_key not in item.assets:
            print(f"  波段 {band_key} 不存在")
            return False
        
        href = item.assets[band_key].href
        
        print(f"  下载波段 {band_key} ...")
        
        # 使用 rioxarray 读取并保存
        data = rioxarray.open_rasterio(href).squeeze()
        data.rio.to_raster(output_path)
        
        print(f"  ✅ 完成: {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        print(f"  ❌ 失败: {band_key} - {e}")
        return False


def download_rgb_composite(item, output_dir, collection="Sentinel-2", scale_factor=0.3):
    """
    下载 RGB 合成图（降采样以减小文件大小）
    
    参数:
        item: STAC Item
        output_dir: 输出目录
        collection: 数据集
        scale_factor: 缩放比例（0.3=30%分辨率）
    
    返回:
        bool: 是否成功
    """
    try:
        if collection == "Sentinel-2":
            bands = {"red": "B04", "green": "B03", "blue": "B02"}
        else:
            bands = {"red": "red", "green": "green", "blue": "blue"}
        
        date_str = item.datetime.strftime("%Y%m%d")
        output_path = os.path.join(output_dir, f"{date_str}_RGB.tif")
        
        if os.path.exists(output_path):
            print(f"  已存在，跳过: {os.path.basename(output_path)}")
            return True
        
        print(f"  下载 RGB 合成图 (scale={scale_factor}) ...")
        
        # 读取各波段
        red = rioxarray.open_rasterio(item.assets[bands["red"]].href).squeeze()
        green = rioxarray.open_rasterio(item.assets[bands["green"]].href).squeeze()
        blue = rioxarray.open_rasterio(item.assets[bands["blue"]].href).squeeze()
        
        # 降采样
        if scale_factor < 1.0:
            new_width = int(red.rio.width * scale_factor)
            new_height = int(red.rio.height * scale_factor)
            red = red.rio.reproject(red.rio.crs, shape=(new_height, new_width), resampling=1)
            green = green.rio.reproject(green.rio.crs, shape=(new_height, new_width), resampling=1)
            blue = blue.rio.reproject(blue.rio.crs, shape=(new_height, new_width), resampling=1)
        
        # 归一化
        def normalize(arr):
            arr = arr.values.astype(np.float32)
            p2, p98 = np.percentile(arr, [2, 98])
            arr = np.clip((arr - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
            return arr
        
        # 创建 RGB 数组
        rgb = np.stack([normalize(red), normalize(green), normalize(blue)], axis=0)
        
        # 保存
        import xarray as xr
        rgb_da = xr.DataArray(rgb, dims=["band", "y", "x"])
        rgb_da.rio.write_crs(red.rio.crs, inplace=True)
        rgb_da.rio.write_transform(red.rio.transform(), inplace=True)
        rgb_da.rio.to_raster(output_path)
        
        # 同时保存预览图
        preview_path = os.path.join(output_dir, f"{date_str}_RGB_preview.jpg")
        Image.fromarray(np.transpose(rgb, (1, 2, 0))).save(preview_path, quality=85)
        
        print(f"  ✅ 完成: {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        print(f"  ❌ RGB合成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_all_bands(item, output_dir, collection="Sentinel-2"):
    """下载所有常用波段"""
    if collection == "Sentinel-2":
        bands = {
            "B02": "蓝色",
            "B03": "绿色", 
            "B04": "红色",
            "B05": "红边1",
            "B06": "红边2",
            "B07": "红边3",
            "B08": "近红外",
            "B8A": "窄近红外",
            "B11": "短波红外1",
            "B12": "短波红外2"
        }
    else:
        bands = {
            "red": "红色",
            "green": "绿色",
            "blue": "蓝色",
            "nir08": "近红外",
            "swir16": "短波红外1",
            "swir22": "短波红外2"
        }
    
    date_str = item.datetime.strftime("%Y%m%d")
    scene_dir = os.path.join(output_dir, f"{date_str}")
    ensure_dir(scene_dir)
    
    print(f"  下载 {len(bands)} 个波段...")
    
    success_count = 0
    for band_key, band_name in bands.items():
        output_path = os.path.join(scene_dir, f"{band_key}.tif")
        if download_band(item, band_key, output_path, collection):
            success_count += 1
    
    print(f"  ✅ 成功 {success_count}/{len(bands)} 个波段")
    return success_count > 0


def main():
    """主函数"""
    print("=" * 60)
    print("Planetary Computer 卫星数据自动下载")
    print("特点: 无需注册，国内可访问，直接下载 GeoTIFF")
    print("=" * 60)
    print()
    
    # 设置下载目录
    output_dir = os.path.join(DOWNLOAD_DIR, STUDY_AREA_NAME, "PlanetaryComputer")
    ensure_dir(output_dir)
    print(f"📁 下载目录: {os.path.abspath(output_dir)}")
    print()
    
    # 连接目录
    print("🌐 连接 Planetary Computer...")
    catalog = get_catalog()
    print("✅ 连接成功")
    print()
    
    all_results = []
    
    # 搜索 Sentinel-2
    if DOWNLOAD_SENTINEL2:
        print("🔍 搜索 Sentinel-2 L2A...")
        results = search_data(
            catalog, "sentinel-2-l2a", BBOX,
            START_DATE, END_DATE,
            MAX_CLOUD_COVER, MAX_SCENES_PER_SATELLITE
        )
        print(f"  找到 {len(results)} 景 Sentinel-2 影像\n")
        all_results.extend([(r, "Sentinel-2") for r in results])
    
    # 搜索 Landsat
    if DOWNLOAD_LANDSAT8 or DOWNLOAD_LANDSAT9:
        print("🔍 搜索 Landsat-8/9...")
        results = search_data(
            catalog, "landsat-c2-l2", BBOX,
            START_DATE, END_DATE,
            MAX_CLOUD_COVER, MAX_SCENES_PER_SATELLITE
        )
        print(f"  找到 {len(results)} 景 Landsat 影像\n")
        all_results.extend([(r, "Landsat") for r in results])
    
    if not all_results:
        print("⚠️  未找到符合条件的影像")
        return
    
    print(f"📊 总计: {len(all_results)} 景影像\n")
    
    # 确认下载
    confirm = input("确认开始下载? (y/n): ").lower()
    if confirm != 'y':
        print("已取消下载")
        return
    
    # 下载模式选择
    print("\n选择下载模式:")
    print("  1. 仅下载 RGB 预览图（文件小，适合快速浏览）")
    print("  2. 下载 RGB + 所有单波段（完整数据，文件大）")
    mode = input("请选择 (1/2): ").strip()
    
    # 开始下载
    print("\n" + "=" * 60)
    print("开始下载...")
    print("=" * 60 + "\n")
    
    success_count = 0
    fail_count = 0
    
    for i, (result, collection) in enumerate(all_results, 1):
        print(f"[{i}/{len(all_results)}] {result['datetime']} | {collection} | 云量: {result['cloud_cover']}%")
        print(f"  ID: {result['id']}")
        
        item = result['item']
        
        if mode == "2":
            # 下载所有波段
            if download_all_bands(item, output_dir, collection):
                success_count += 1
            else:
                fail_count += 1
        else:
            # 仅下载 RGB
            if download_rgb_composite(item, output_dir, collection, scale_factor=0.3):
                success_count += 1
            else:
                fail_count += 1
        
        print()
        
        # 短暂休息，避免请求过快
        time.sleep(1)
    
    # 统计结果
    print("=" * 60)
    print("下载完成!")
    print(f"✅ 成功: {success_count} 景")
    print(f"❌ 失败: {fail_count} 景")
    print(f"📁 保存位置: {os.path.abspath(output_dir)}")
    print("=" * 60)
    
    print("\n💡 提示:")
    print("  - 下载的是 GeoTIFF 格式，可直接在 ENVI/ArcGIS/QGIS 中打开")
    print("  - RGB 预览图为 .jpg 格式，方便快速浏览")
    print("  - 如要重新下载失败的影像，重新运行脚本即可（已下载的会自动跳过）")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断下载")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
