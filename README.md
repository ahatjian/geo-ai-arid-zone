# 🌍 西北干旱区遥感智能分析平台

> v1.4 | Phase 1-3 全部完成 | Microsoft Planetary Computer + Streamlit + 深度学习

基于 **Microsoft Planetary Computer (STAC API)** + **Streamlit** + **geoai-py 深度学习**的 Web 端干旱区遥感影像智能分析应用。面向中国西北干旱半干旱地区，提供卫星数据检索/预览、植被/水体指数分析、AI地物分类、变化检测及报告自动导出全流程功能。

## 功能特点

- 🗺️ **STAC 数据浏览** — 通过 Planetary Computer STAC API 搜索 Sentinel-2/Landsat 影像，支持 RGB/NDVI/MNDWI 实时预览
- 💧 **水体动态监测** — MNDWI/AWEIsh 指数分析 + OmniWaterMask (UNet+ResNet34) AI 水体分割
- 🌿 **植被指数分析** — NDVI/EVI 时序监测 + Sen+MK 趋势分析（季节性 Theil-Sen + Mann-Kendall）
- 🤖 **AI 地物分类** — ESA WorldCover / ESRI Land Cover 公开产品优先 + 深度学习推理，覆盖6类土地覆盖（水体/植被/裸地/建设用地/农田/矿区）
- 🔄 **变化检测** — 双时相7级变化分类（重度增加/中度增加/轻度增加/未变化/轻度减少/中度减少/重度减少）
- 📊 **报告导出** — 一键生成多页 HTML 分析报告，自动采集 session_state 数据
- ⚡ **ONNX 推理加速** — ONNX Runtime CPU 加速，覆盖水体分割和AI分类两个页面

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端框架 | Streamlit (多页面应用) |
| 地图渲染 | leafmap + Folium |
| 数据源 | **Microsoft Planetary Computer** (STAC API, 免费, 国内可访问) |
| 卫星数据 | Sentinel-2, Landsat 8/9 |
| AI 引擎 | PyTorch + **geoai-py** (UNet / DeepLabV3+) |
| 推理加速 | **ONNX Runtime** v1.20.1 (CPUExecutionProvider) |
| 地理处理 | rasterio / GDAL / shapely |
| 统计分析 | Sen+MK (pymannkendall 1.4.3) |
| 部署 | Streamlit Community Cloud (免费) |

## 快速开始

### 1. 环境配置

```bash
# 创建 Conda 环境 (推荐 Python 3.11)
conda create -n geo-ai python=3.11 -y
conda activate geo-ai

# 安装依赖
cd web-geo-ai
pip install -r requirements.txt
```

### 2. 预下载 AI 模型

```bash
# 首次运行时会自动下载 OmniWaterMask 预训练权重
# 或手动提前下载以加速冷启动
python -c "from geoai import segment_water; print('Models ready')"
```

### 3. 运行应用

```bash
streamlit run app.py
```

应用将在本地启动，默认访问 http://localhost:8501

## 项目结构

```
web-geo-ai/
├── app.py                        # 首页入口 + 6研究区预设
├── pages/                        # 多页面（Streamlit 自动路由）
│   ├── 1_数据浏览.py              # STAC搜索 + RGB/NDVI/MNDWI预览
│   ├── 2_水体监测.py              # MNDWI/AWEIsh + AI水体分割(含ONNX)
│   ├── 3_植被分析.py              # NDVI/EVI + Sen+MK趋势分析
│   ├── 4_AI分类.py               # ESA/ESRI公开产品 + AI推理(含ONNX)
│   ├── 5_变化检测.py              # 双时相7级变化分类
│   └── 6_报告导出.py              # HTML报告自动生成
├── utils/                        # 工具函数库
│   ├── pc_data.py                # Planetary Computer STAC 数据获取与缓存
│   ├── indices.py                # 遥感指数计算 (NDVI/MNDWI/EVI/AWEIsh)
│   ├── ai_engine.py              # geoai-py AI推理引擎
│   ├── landcover.py              # ESA/ESRI 土地覆盖数据
│   ├── visualization.py          # 影像可视化与地图渲染
│   ├── trend.py                  # Sen+MK 趋势分析
│   ├── export.py                 # HTML报告导出
│   ├── onnx_engine.py            # ONNX Runtime 推理加速 (784行, 8函数)
│   └── error_handler.py          # 统一错误处理与用户提示
├── config.py                     # 全局配置（缓存TTL/ONNX参数/研究区BBOX等）
├── models/                       # 本地AI模型存储
├── data/                         # 下载数据缓存
├── scripts/                      # 辅助脚本
├── docs/                         # 项目文档
├── .streamlit/
│   ├── config.toml               # Streamlit 主题/服务器配置
│   └── secrets.toml.template     # 密钥模板
├── .gitignore                    # Git 忽略规则
├── Dockerfile                    # Docker 容器化部署
├── packages.txt                  # Streamlit Cloud 系统依赖
├── runtime.txt                   # Streamlit Cloud Python 版本
├── requirements.txt              # Python 依赖
└── README.md
```

## 研究区域

6 个预设研究区，覆盖中国西北干旱半干旱核心区域：

| 编号 | 研究区 | BBOX [lon_min, lat_min, lon_max, lat_max] |
|------|--------|------|
| 1 | 塔里木盆地 | [76, 37, 88, 42] |
| 2 | 柴达木盆地 | [94, 36, 98, 39] |
| 3 | 河西走廊 | [96, 38, 104, 42] |
| 4 | 吐鲁番盆地 | [88, 42, 90, 43.5] |
| 5 | 天山北坡 | [82, 43, 90, 45] |
| 6 | 准噶尔盆地 | [82, 44, 92, 48] |

## AI 模型

| 模型 | 架构 | 预训练 | 功能 |
|------|------|--------|------|
| OmniWaterMask | UNet + ResNet34 | geoai 预训练 | AI 水体分割 |
| ESA WorldCover | 公开产品 | ESA | 10m 全球土地覆盖 |
| ESRI Land Cover | 公开产品 | ESRI | 10m 全球土地覆盖 |
| DeepLabV3+ | ResNet50 主干 | ImageNet | 自定义地物分类 |

> **ONNX 加速**: 水体分割和AI分类支持 PyTorch ↔ ONNX 双推理模式，CPU 推理速度提升 2-5x。

## 缓存架构

- 条件缓存装饰器 `_cache(ttl)` 在非 Streamlit 环境自动降级为 no-op
- STAC Item 对象绕过不可哈希限制
- 三级 TTL 配置（统一管理于 config.py）:
  - `ttl_short` (600s) ×2: 数据搜索/目录查询
  - `ttl_medium` (1800s) ×5: 影像预览/指数计算
  - `ttl_long` (7200s) ×3: AI推理/分类结果

## 数据来源

- [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) — STAC API 卫星影像数据
- [Sentinel-2](https://sentinel.esa.int/) — 欧空局哨兵2号 (10m 分辨率)
- [Landsat 8/9](https://landsat.gsfc.nasa.gov/) — 美国陆地卫星 (30m 分辨率)
- [ESA WorldCover](https://worldcover2021.esa.int/) — 全球10m土地覆盖产品
- [ESRI Land Cover](https://livingatlas.arcgis.com/landcover/) — ESRI 全球10m土地覆盖

## 部署

### Streamlit Community Cloud (免费, 推荐)

1. 将代码推送到 GitHub
2. 访问 [share.streamlit.io](https://share.streamlit.io)
3. 连接 GitHub 仓库，主文件设为 `app.py`
4. 选择对应分支部署

### Docker 部署

```bash
docker build -t geo-ai-app .
docker run -p 8501:8501 geo-ai-app
```

## 环境状态

- **Python**: 3.11 | **PyTorch**: 2.11.0+cpu
- **GPU**: 当前无 NVIDIA GPU，训练需 GPU 环境
- **推理**: ONNX Runtime CPU 模式已优化

## License

MIT License

---

**项目状态**: v1.4 — 所有 3 个 Phase 全部完成，6 个功能页面均已实现，代码审查优化项全部落实 ✅
