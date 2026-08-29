"""
Geo AI 干旱区遥感分析平台 — 全局配置
统一管理研究区、卫星数据源、模型参数等配置项
"""

import os

# ============================================
# 平台信息
# ============================================
APP_TITLE = "西北干旱区 Geo AI 智能分析平台"
APP_ICON = "🛰️"
APP_VERSION = "v1.12"
APP_DESCRIPTION = "免费卫星数据 + AI 自动分析 = 不写代码做科研级遥感分析"

# ============================================
# 研究区域 (6 大干旱区核心区域)
# ============================================
STUDY_AREAS = {
    "塔里木盆地": {
        "bbox": [76, 37, 88, 42],
        "center": [39.5, 82],
        "description": "中国最大内陆盆地，沙漠绿洲交错",
        "keywords": ["沙漠", "绿洲", "塔里木河"]
    },
    "柴达木盆地": {
        "bbox": [94, 36, 98, 39],
        "center": [37.5, 96],
        "description": "盐湖密布，干涸湖泊监测重点区",
        "keywords": ["盐湖", "干涸湖", "钾盐"]
    },
    "河西走廊": {
        "bbox": [96, 38, 104, 42],
        "center": [40, 100],
        "description": "绿洲农业走廊，水资源敏感区",
        "keywords": ["绿洲", "农业", "祁连山融水"]
    },
    "吐鲁番盆地": {
        "bbox": [88, 42, 90, 43.5],
        "center": [42.7, 89],
        "description": "极端干旱，艾丁湖为中国最低点",
        "keywords": ["极端干旱", "艾丁湖", "葡萄沟"]
    },
    "天山北坡": {
        "bbox": [82, 43, 90, 45],
        "center": [44, 86],
        "description": "冰川融水补给绿洲，水资源关键区",
        "keywords": ["冰川", "融水", "绿洲"]
    },
    "准噶尔盆地": {
        "bbox": [82, 44, 92, 48],
        "center": [46, 87],
        "description": "荒漠-草原过渡带，生态脆弱区",
        "keywords": ["荒漠草原", "古尔班通古特", "过渡带"]
    },
}

# ============================================
# Planetary Computer 卫星数据配置
# ============================================
COLLECTIONS = {
    "Sentinel-2 L2A": {
        "id": "sentinel-2-l2a",
        "resolution": 10,
        "bands": {
            "blue": "B02",
            "green": "B03",
            "red": "B04",
            "nir": "B08",
            "swir1": "B11",
            "swir2": "B12",
        },
        "rgb_bands": ["B04", "B03", "B02"],
        "description": "欧空局哨兵2号，10m分辨率，5天重访"
    },
    "Landsat-8": {
        "id": "landsat-c2-l2",
        "resolution": 30,
        "bands": {
            "blue": "SR_B2",
            "green": "SR_B3",
            "red": "SR_B4",
            "nir": "SR_B5",
            "swir1": "SR_B6",
            "swir2": "SR_B7",
        },
        "rgb_bands": ["SR_B4", "SR_B3", "SR_B2"],
        "description": "USGS Landsat-8，30m分辨率，16天重访 (2013-至今)",
        "temporal_range": "2013-至今",
    },
    "Landsat-9": {
        "id": "landsat-c2-l2",
        "resolution": 30,
        "bands": {
            "blue": "SR_B2",
            "green": "SR_B3",
            "red": "SR_B4",
            "nir": "SR_B5",
            "swir1": "SR_B6",
            "swir2": "SR_B7",
        },
        "rgb_bands": ["SR_B4", "SR_B3", "SR_B2"],
        "description": "USGS Landsat-9，30m分辨率，16天重访 (2021-至今)",
        "temporal_range": "2021-至今",
    },
    "Landsat-7": {
        "id": "landsat-c2-l2",
        "resolution": 30,
        "bands": {
            "blue": "SR_B2",
            "green": "SR_B3",
            "red": "SR_B4",
            "nir": "SR_B5",
            "swir1": "SR_B6",
            "swir2": "SR_B7",
        },
        "rgb_bands": ["SR_B4", "SR_B3", "SR_B2"],
        "description": "USGS Landsat-7，30m分辨率，16天重访 (1999-至今，有条带)",
        "temporal_range": "1999-至今",
    },
    "Landsat-4-5": {
        "id": "landsat-c2-l2",
        "resolution": 30,
        "bands": {
            "blue": "SR_B2",
            "green": "SR_B3",
            "red": "SR_B4",
            "nir": "SR_B5",
            "swir1": "SR_B6",
            "swir2": "SR_B7",
        },
        "rgb_bands": ["SR_B4", "SR_B3", "SR_B2"],
        "description": "USGS Landsat 4-5，30m分辨率，16天重访 (1982-2013)",
        "temporal_range": "1982-2013",
    },
}

# ============================================
# Landsat 地表温度 (ST) 产品配置
# ============================================
# USGS Collection 2 Level-2 地表温度产品资产名 (L8/9 为 ST_B10, L4/5/7 为 ST_B6)
LANDSAT_ST_ASSET = {
    "Landsat-8": "ST_B10",
    "Landsat-9": "ST_B10",
    "Landsat-7": "ST_B6",
    "Landsat-4-5": "ST_B6",
}
# ST 产品辐射定标: LST_Kelvin = DN * scale + offset
LANDSAT_ST_SCALE = 0.00341802
LANDSAT_ST_OFFSET = 149.0  # Kelvin

# ============================================================
# Landsat 长时序说明
# ============================================================
# Planetary Computer 的 landsat-c2-l2 涵盖 Landsat 4-9 全部 Level-2 数据
# 最长时序: Landsat-4-5 (1982) 至今 ~40年 NDVI 重建
# 推荐使用 Landsat 4-5/7/8/9 组合进行长时序分析
# STAC 筛选: platform=landsat-5 / landsat-7 / landsat-8 / landsat-9

# ============================================
# 干旱指数阈值配置
# ============================================
DROUGHT_INDEX_THRESHOLDS = {
    "SPI": {
        "extreme_dry": -2.0, "severe_dry": -1.5, "moderate_dry": -1.0,
        "normal": 1.0, "moderate_wet": 1.5, "extreme_wet": 2.0,
    },
    "VCI": {
        "extreme_dry": 10, "severe_dry": 20, "moderate_dry": 30,
        "normal": 40, "good": 50, "excellent": 70,
    },
    "NDDI": {
        "drought": 0.5, "severe_drought": 0.7,
    },
    "TVDI": {
        "wet": 0.2, "normal": 0.4, "dry": 0.6, "very_dry": 0.8,
    },
    "NDVI_ANOMALY": {
        "severe_decrease": -2.0, "moderate_decrease": -1.0,
        "normal": 1.0, "increase": 2.0,
    },
}

# ============================================
# 遥感指数阈值配置
# ============================================
INDEX_THRESHOLDS = {
    "MNDWI": {"water": 0.0, "description": "MNDWI > 0 为水体"},
    "AWEIsh": {"water": 0.0, "description": "AWEIsh > 0 为水体"},
    "NDVI": {
        "dense_veg": 0.6,
        "sparse_veg": 0.2,
        "bare": 0.1,
        "description": "<0.1 裸地/水体, 0.2-0.4 草地, >0.6 森林"
    },
    "EVI": {
        "dense_veg": 0.4,
        "sparse_veg": 0.15,
        "description": "EVI 避免饱和，更适合密植被区"
    },
}

# ============================================
# AI 模型配置
# ============================================
AI_MODELS = {
    "water_seg": {
        "name": "水体分割模型",
        "architectures": ["UNet"],
        "encoders": ["resnet34"],
        "num_classes": 2,
        "class_names": ["背景", "水体"],
        "class_colors": ["#444444", "#0066FF"],
        "input_bands": ["blue", "green", "red", "nir", "swir1", "swir2"],
        "tile_size": 512,
        "overlap": 256,
    },
    "land_cover": {
        "name": "土地覆盖分类模型",
        "architectures": ["UNet", "DeepLabV3+", "FPN"],
        "encoders": ["resnet50", "resnet34"],
        "num_classes": 6,
        "class_names": ["水体", "植被", "裸地", "建设用地", "农田", "矿区"],
        "class_colors": ["#0066FF", "#00CC44", "#CCCCCC", "#FF4444", "#FFAA00", "#8B4513"],
        "input_bands": ["blue", "green", "red", "nir", "swir1", "swir2"],
        "tile_size": 512,
        "overlap": 256,
    },
}

# ============================================
# 路径配置
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, ".cache")

# 确保目录存在
for d in [MODELS_DIR, DATA_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================
# ONNX 推理加速配置
# ============================================
ONNX_CONFIG = {
    "enabled": True,
    "window_size": 512,        # 滑动窗口大小 (像素)
    "overlap": 256,            # 窗口重叠大小
    "batch_size": 4,           # 推理批次大小
    "opset_version": 13,       # ONNX opset 版本
    "simplify": True,          # 是否简化模型
    "providers": ["CPUExecutionProvider"],  # 默认 CPU (GPU: CUDAExecutionProvider)
}

# 预训练 ONNX 模型路径 (放入 models/ 目录)
ONNX_MODELS = {
    "water_seg": "models/water_seg_unet_resnet34.onnx",
    "land_cover": "models/landcover_unet_resnet50.onnx",
}

# ============================================
# 可视化色带配置
# ============================================
COLORMAPS = {
    "NDVI": "RdYlGn",
    "MNDWI": "Blues",
    "AWEIsh": "coolwarm",
    "EVI": "YlGn",
    "water_mask": "Blues",
    "land_cover": "Set1",
    "change": "RdYlGn",  # 红=减少, 绿=增加, 黄=不变
    # 干旱指数专用色带
    "SPI": "RdYlBu",           # 红(干) ↔ 蓝(湿)
    "SPEI": "RdYlBu",
    "VCI": "YlOrRd_r",         # 反转: 绿色(好) → 红色(干)
    "VHI": "YlOrRd_r",
    "NDDI": "YlOrRd",          # 红=干旱
    "TVDI": "YlOrRd",
    "NDVI_ANOMALY": "RdBu_r",  # 蓝(增加) ↔ 红(减少)
    "CDI": "RdYlGn",           # 红(干) ↔ 绿(湿)
    "DROUGHT_CATEGORY": "YlOrRd",
}

# ============================================
# 性能与缓存配置
# ============================================
CACHE_CONFIG = {
    "ttl_short": 600,           # 短缓存: 10 分钟 (指数计算结果)
    "ttl_medium": 1800,         # 中缓存: 30 分钟 (STAC 搜索结果)
    "ttl_long": 7200,           # 长缓存: 2 小时 (土地覆盖数据)
    "max_entries": 50,          # 最大缓存条目数
    "show_spinner": True,       # 显示加载动画
}

# ============================================
# 错误处理与重试配置
# ============================================
RETRY_CONFIG = {
    "max_retries": 3,           # 最大重试次数
    "backoff_factor": 2,        # 退避因子 (指数退避)
    "initial_delay": 2,         # 初始延迟 (秒)
    "timeout_http": 60,         # HTTP 请求超时 (秒)
    "timeout_stac": 120,        # STAC 搜索超时 (秒)
}

# ============================================
# 部署模式
# ============================================
# "local": 本地开发 | "streamlit_cloud": Streamlit Community Cloud | "docker": Docker 部署
DEPLOY_MODE = "local"
