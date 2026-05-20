"""
Landsat 卫星数据自动下载脚本
支持 Landsat-8/9 Collection 2 Level-2 产品
使用 USGS Earth Explorer API

使用方法:
1. 在 config.py 中填写 USGS 账号密码
2. 修改研究区、时间范围等参数
3. 运行: python download_landsat.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# 自动安装所需库
required_packages = ['landsatxplore', 'tqdm', 'requests']
for package in required_packages:
    try:
        __import__(package.replace('-', '_'))
    except ImportError:
        print(f"正在安装 {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple'])

from landsatxplore.api import API
from landsatxplore.earthexplorer import EarthExplorer
from tqdm import tqdm

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


def search_landsat_data(username, password, bbox, start_date, end_date, max_cloud=20, max_results=50):
    """
    搜索 Landsat 影像
    
    参数:
        username: USGS 用户名
        password: USGS 密码
        bbox: [min_lon, min_lat, max_lon, max_lat]
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        max_cloud: 最大云量 (%)
        max_results: 最大返回数量
    
    返回:
        list: 影像列表
    """
    api = API(username, password)
    
    datasets = []
    if DOWNLOAD_LANDSAT8:
        datasets.append('landsat_ot_c2_l2')  # Landsat-8 Collection 2 Level-2
    if DOWNLOAD_LANDSAT9:
        datasets.append('landsat_ot_c2_l2')  # Landsat-9 和 Landsat-8 用同一个数据集
    
    all_scenes = []
    
    for dataset in datasets:
        print(f"正在搜索 {dataset} ...")
        try:
            scenes = api.search(
                dataset=dataset,
                bbox=bbox,
                start_date=start_date,
                end_date=end_date,
                max_cloud_cover=max_cloud,
                max_results=max_results
            )
            print(f"  找到 {len(scenes)} 景 {dataset} 影像")
            all_scenes.extend(scenes)
        except Exception as e:
            print(f"  搜索 {dataset} 失败: {e}")
    
    api.logout()
    return all_scenes


def download_scene(ee, scene_id, output_dir, timeout=600):
    """
    下载单景影像
    
    参数:
        ee: EarthExplorer 实例
        scene_id: 影像ID
        output_dir: 输出目录
        timeout: 超时时间
    
    返回:
        bool: 是否成功
    """
    try:
        output_path = os.path.join(output_dir, f"{scene_id}.tar")
        
        # 检查是否已下载
        if os.path.exists(output_path):
            print(f"  已存在，跳过: {scene_id}")
            return True
        
        print(f"  正在下载: {scene_id} ...")
        ee.download(scene_id, output_dir=output_dir, timeout=timeout)
        print(f"  ✅ 下载完成: {scene_id}")
        return True
        
    except Exception as e:
        print(f"  ❌ 下载失败: {scene_id} - {str(e)}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("Landsat 卫星数据自动下载工具")
    print("=" * 60)
    print()
    
    # 检查配置
    if not USGS_USERNAME or not USGS_PASSWORD:
        print("❌ 错误: 请在 config.py 中填写 USGS 账号和密码")
        print("  注册地址: https://earthexplorer.usgs.gov/")
        return
    
    if not DOWNLOAD_LANDSAT8 and not DOWNLOAD_LANDSAT9:
        print("⚠️  config.py 中 Landsat 下载已关闭，跳过")
        return
    
    # 设置下载目录
    if ORGANIZE_BY_YEAR:
        output_dir = os.path.join(DOWNLOAD_DIR, STUDY_AREA_NAME, "Landsat")
    else:
        output_dir = os.path.join(DOWNLOAD_DIR, STUDY_AREA_NAME)
    
    ensure_dir(output_dir)
    print(f"📁 下载目录: {os.path.abspath(output_dir)}")
    print()
    
    # 搜索数据
    print(f"🔍 搜索参数:")
    print(f"  研究区: {STUDY_AREA_NAME}")
    print(f"  边界框: {BBOX}")
    print(f"  时间范围: {START_DATE} 至 {END_DATE}")
    print(f"  最大云量: {MAX_CLOUD_COVER}%")
    print()
    
    scenes = search_landsat_data(
        USGS_USERNAME,
        USGS_PASSWORD,
        BBOX,
        START_DATE,
        END_DATE,
        MAX_CLOUD_COVER,
        MAX_SCENES_PER_SATELLITE
    )
    
    if not scenes:
        print("⚠️  未找到符合条件的影像，尝试放宽条件（增大云量限制或时间范围）")
        return
    
    print(f"\n📊 共找到 {len(scenes)} 景影像，开始下载...")
    print()
    
    # 登录 EarthExplorer 进行下载
    ee = EarthExplorer(USGS_USERNAME, USGS_PASSWORD)
    
    # 下载所有影像
    success_count = 0
    fail_count = 0
    
    for i, scene in enumerate(scenes, 1):
        scene_id = scene['display_id']
        print(f"[{i}/{len(scenes)}] {scene_id}")
        print(f"  日期: {scene.get('acquisition_date', 'N/A')}")
        print(f"  云量: {scene.get('cloud_cover', 'N/A')}%")
        
        if download_scene(ee, scene_id, output_dir, DOWNLOAD_TIMEOUT):
            success_count += 1
        else:
            fail_count += 1
            if fail_count >= MAX_RETRIES:
                print(f"  ⚠️  连续失败 {MAX_RETRIES} 次，暂停后继续...")
                time.sleep(60)
        
        print()
    
    ee.logout()
    
    # 统计结果
    print("=" * 60)
    print("下载完成!")
    print(f"✅ 成功: {success_count} 景")
    print(f"❌ 失败: {fail_count} 景")
    print(f"📁 保存位置: {os.path.abspath(output_dir)}")
    print("=" * 60)
    
    # 提示解压
    print("\n💡 提示: 下载的文件是 .tar 格式，可以用 7-Zip 或 WinRAR 解压")
    print("   解压后会得到 GeoTIFF (.tif) 文件，可直接在 ENVI/ArcGIS 中打开")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断下载")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
