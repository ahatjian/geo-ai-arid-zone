"""
卫星数据批量下载统一调度脚本
支持多种数据源，自动选择最优方案

使用方法:
1. 修改 config.py 中的参数
2. 运行: python download_all.py
3. 按提示选择下载源
"""

import os
import sys
import subprocess


def check_install(package):
    """检查并安装包"""
    try:
        __import__(package.replace('-', '_'))
        return True
    except ImportError:
        print(f"正在安装 {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple'])
        return True


def show_menu():
    """显示菜单"""
    print("\n" + "=" * 60)
    print("🛰️  卫星数据自动下载工具")
    print("=" * 60)
    print()
    print("请选择下载源:")
    print()
    print("  [1] Planetary Computer (推荐)")
    print("      优点: 无需注册账号，国内可访问，直接下载 GeoTIFF")
    print("      缺点: 下载速度一般")
    print()
    print("  [2] USGS Earth Explorer (Landsat)")
    print("      优点: 数据最全，历史最长 (1984-至今)")
    print("      缺点: 需要 USGS 账号，国内访问较慢")
    print()
    print("  [3] Copernicus Hub (Sentinel-2)")
    print("      优点: Sentinel-2 数据质量最高 (10m)")
    print("      缺点: 需要 Copernicus 账号，国内访问较慢")
    print()
    print("  [0] 退出")
    print()
    print("=" * 60)


def main():
    """主函数"""
    show_menu()
    
    choice = input("请输入选项 (0/1/2/3): ").strip()
    
    if choice == "1":
        print("\n🚀 启动 Planetary Computer 下载...")
        print("特点: 无需注册，直接下载\n")
        os.system(f"{sys.executable} download_planetary.py")
        
    elif choice == "2":
        print("\n🚀 启动 USGS Landsat 下载...")
        print("请先确保已在 config.py 中填写 USGS 账号\n")
        os.system(f"{sys.executable} download_landsat.py")
        
    elif choice == "3":
        print("\n🚀 启动 Copernicus Sentinel-2 下载...")
        print("请先确保已在 config.py 中填写 Copernicus 账号\n")
        os.system(f"{sys.executable} download_sentinel.py")
        
    elif choice == "0":
        print("退出")
        return
    else:
        print("无效选项")
        return
    
    print("\n" + "=" * 60)
    print("所有任务已完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
