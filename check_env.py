"""
环境诊断脚本 - 完整健康检查
==============================
检查: Python 环境 / 核心依赖 / DeepSeek API / STAC 数据源 / 关键目录
用于部署前或演示前快速确认平台可运行。

用法:
    python check_env.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "✅"
WARN = "⚠️"
FAIL = "❌"

# 核心依赖 (按重要性排序)
CORE_DEPS = [
    "streamlit", "numpy", "pandas", "rasterio", "geopandas",
    "shapely", "requests", "matplotlib", "plotly",
]

# 功能依赖 (缺失时部分功能不可用, 平台仍可启动)
# 键为导入名 (import 语句使用的名字), 值为用途说明
FUNC_DEPS = {
    "leafmap": "研究区概览地图",
    "pystac_client": "STAC 影像搜索",
    "planetary_computer": "PC 数据签名",
    "xarray": "栅格数据处理",
    "rioxarray": "栅格数据处理",
    "statsmodels": "干旱预测 (SARIMA/Holt-Winters)",
    "sklearn": "监督分类/预测",
    "pymannkendall": "Sen+MK 趋势分析",
    "onnxruntime": "ONNX 推理加速",
    "torch": "深度学习推理 (CPU)",
    "torchvision": "深度学习推理",
    "segmentation_models_pytorch": "AI 分割模型",
    "PIL": "图像处理",
    "imageio": "GIF 动画",
    "seaborn": "统计可视化",
    "scipy": "科学计算",
    "fiona": "矢量读写",
    "pyproj": "坐标转换",
    "cv2": "图像处理",
}


def check_import(mod_name: str) -> bool:
    try:
        __import__(mod_name)
        return True
    except ImportError:
        return False


def main() -> int:
    print("=" * 56)
    print("Geo AI 遥感分析平台 - 环境健康检查")
    print("=" * 56)

    exit_code = 0

    # ---- 1. Python 环境 ----
    print(f"\n[1/5] Python 环境")
    print(f"  Python: {sys.version.split()[0]} ({sys.executable})")

    # ---- 2. 核心依赖 ----
    print(f"\n[2/5] 核心依赖")
    missing_core = []
    for dep in CORE_DEPS:
        if check_import(dep):
            print(f"  {PASS} {dep}")
        else:
            print(f"  {FAIL} {dep}")
            missing_core.append(dep)
    if missing_core:
        exit_code = 1

    print(f"\n  [功能依赖] (缺失仅影响对应功能)")
    for dep, purpose in FUNC_DEPS.items():
        if check_import(dep):
            print(f"  {PASS} {dep} ({purpose})")
        else:
            print(f"  {WARN} {dep} ({purpose})")

    # ---- 3. DeepSeek API ----
    print(f"\n[3/5] DeepSeek API")
    api_key = ""
    try:
        import streamlit as st
        api_key = st.secrets.get("DEEPSEEK_API_KEY", "")
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if api_key:
        print(f"  {PASS} DEEPSEEK_API_KEY 已配置 (智能查询 + AI 解读可用)")
    else:
        print(f"  {WARN} 未配置 DEEPSEEK_API_KEY — 将使用模板匹配/规则解读")
        print(f"       配置方法: cp .streamlit/secrets.toml.template .streamlit/secrets.toml")
        print(f"       并在 secrets.toml 填入真实 Key")

    # ---- 4. STAC 数据源 ----
    print(f"\n[4/5] Planetary Computer STAC 数据源")
    if check_import("pystac_client"):
        try:
            from pystac_client import Client
            catalog = Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                timeout=15,
            )
            print(f"  {PASS} STAC 连接成功 ({catalog.title})")
        except Exception as e:
            print(f"  {FAIL} STAC 连接失败: {type(e).__name__}: {str(e)[:120]}")
            exit_code = 1
    else:
        print(f"  {WARN} pystac_client 未安装, 跳过 STAC 检查")

    # ---- 5. 关键目录 ----
    print(f"\n[5/5] 关键目录")
    for d in ["data", "models", ".cache", "results"]:
        os.makedirs(d, exist_ok=True)
        print(f"  {PASS} {d}/ 就绪")

    print("\n" + "=" * 56)
    if exit_code == 0:
        print("结论: 环境就绪, 可以启动平台 (streamlit run app.py)")
    else:
        print("结论: 存在缺失核心依赖, 请先 pip install -r requirements.txt")
    print("=" * 56)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
