# 🌍 西北干旱区遥感智能分析平台

> v1.15 | Phase 1-6 全部完成 | 20 分析模块 + DeepSeek AI 智能查询与解读 + 自定义 AOI + 矢量导出 + 数据下载中心

基于 **Microsoft Planetary Computer (STAC API)** + **Streamlit** + **DeepSeek AI** 的 Web 端干旱区遥感智能分析应用。面向中国西北干旱半干旱地区，提供卫星数据检索、遥感指数计算、AI地物分类、干旱预测、沙漠化评估、冰冻圈分析、农业干旱监测、时序动画及智能工作流全链路功能。

## 功能特点

### 🏠 基础分析 (4 模块)
- 🗺️ **数据浏览** — 通过 Planetary Computer STAC API 搜索 Sentinel-2/Landsat 4-9 影像，支持 RGB/NDVI/MNDWI 实时预览
- 💧 **水体监测** — MNDWI/AWEIsh 指数分析 + OmniWaterMask (UNet+ResNet34) AI 水体分割 + ONNX 加速
- 🌿 **植被分析** — NDVI/EVI 时序监测 + Sen+MK 趋势分析（Theil-Sen + Mann-Kendall）
- 🤖 **AI 分类** — ESA WorldCover / ESRI Land Cover 公开产品 + 深度学习推理 (PyTorch ↔ ONNX 双模式)

### 🔬 专业分析 (10 模块)
- 🏜️ **干旱监测** — 8 种干旱指数 (VCI/NDDI/SPI/SPEI/TVDI/CDI) + SARIMA/LSTM/Holt-Winters 预测 + Albedo-NDVI 沙漠化 5 级评估
- ❄️ **冰冻圈分析** — NDSI 积雪 4 级分类 + NIR/SWIR 冰川边界提取（积累区/消融区）+ 冻土活动层分析
- 🌾 **农业干旱** — CWSI 作物水分胁迫 + SMI 土壤水分 + MPDI 垂直干旱指数 + 灌溉需求评估
- 🔄 **变化检测** — 双时相 7 级变化分类 + 统计报告
- 🧂 **土壤盐渍化** — SI/SI1/SI2/NDSI/BI 盐分指数 + 5 级盐渍化评估（植被/水体掩膜综合分级）
- 🌡️ **地表温度 LST** — Landsat 热红外 (ST_B10/ST_B6) 地表温度反演 + 5 级热环境分级 + LST-NDVI 关系
- 💨 **蒸散发估算** — SEBAL 简化能量平衡法 (Rn=H+LE+G) 地表蒸散发 ET + 5 级分级 + 能量分量
- 🎯 **监督分类** — 自定义样本训练模型 (随机森林/SVM/KNN/MLP) + 混淆矩阵/OA/Kappa 精度评估
- 🧮 **指数计算器** — 12 种预设指数 + 自定义波段运算 (Band Math) 表达式
- 🔀 **土地转移矩阵** — 双时相 ESA WorldCover 对比 + 转移矩阵/净变化/主要转移方向

### ✨ 智能化 (6 模块)
- 📄 **报告导出** — 一键生成多页 HTML 综合分析报告
- 🎬 **时序动画** — NDVI/水体/雪盖 GIF 动画生成 (单指数/多指数对比/趋势曲线 3 种模式)
- 🌍 **生态评估** — PSR 压力-状态-响应 生态安全指数 (ESI) + 可调权重 + 5 级安全分级
- ⚡ **智能工作流** — DeepSeek AI 自然语言查询 + 一键分步分析 + 综合报告自动生成 + AI 智能解读
- 🤖 **AI 智能解读** — 基于各模块分析指标，DeepSeek 自动生成专业生态/环境解读，嵌入 HTML 报告（无 API Key 时自动降级为规则模板）
- 🗺️ **矢量导出** — 分类/分级/掩膜结果矢量化，导出 GeoJSON / Shapefile (zip) / KML，衔接 ArcGIS/QGIS/Google Earth
- 📦 **数据下载中心** — 结果持久化 (跨会话) + 统一下载 / 批量打包 (zip) / 预览 / 删除

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Streamlit (20 页面应用) |
| 地图 | leafmap + Folium |
| 数据源 | **Microsoft Planetary Computer** (STAC API, 免费, 国内可访问) |
| 卫星 | Sentinel-2 (10m, 2015-) + Landsat 4-9 (30m, 1982-, 43年长时序) |
| AI | PyTorch + geoai-py (UNet/DeepLabV3+) + ONNX Runtime 推理加速 |
| 统计 | statsmodels (SARIMA/Holt-Winters) + scikit-learn + pymannkendall |
| 智能 | DeepSeek API (自然语言查询, 可降级为模板匹配) |
| 地理 | rasterio/GDAL/shapely/geopandas |
| 部署 | Streamlit Community Cloud (免费) / Docker |

## 快速开始

```bash
# 环境配置
conda create -n geo-ai python=3.11 -y && conda activate geo-ai
cd web-geo-ai && pip install -r requirements.txt

# 启动应用
streamlit run app.py
# 访问 http://localhost:8501
```

## DeepSeek AI 配置

平台 AI 能力（自然语言查询 + 智能解读）默认启用，配置 API Key 后自动激活：

```bash
# 1. 注册 DeepSeek 开放平台: https://platform.deepseek.com/
# 2. 创建 API Key, 然后配置 (三选一):

# 方式 A: Streamlit secrets (推荐)
cp .streamlit/secrets.toml.template .streamlit/secrets.toml
# 编辑 secrets.toml, 填入 DEEPSEEK_API_KEY = "sk-xxx"

# 方式 B: 环境变量
export DEEPSEEK_API_KEY=sk-xxx

# 方式 C: Streamlit Cloud 部署时在 App Settings → Secrets 中设置
```

未配置 Key 时平台自动降级为关键词模板匹配/规则解读，功能不受影响。

## 项目结构

```
web-geo-ai/
├── app.py                        # 首页入口 + 20模块导航
├── pages/
│   ├── 1_数据浏览.py              # STAC搜索 + RGB/NDVI/MNDWI预览
│   ├── 2_水体监测.py              # MNDWI/AWEIsh + AI水体分割(含ONNX)
│   ├── 3_植被分析.py              # NDVI/EVI + Sen+MK趋势
│   ├── 4_AI分类.py               # ESA/ESRI + AI推理(含ONNX)
│   ├── 5_变化检测.py              # 双时相7级变化分类
│   ├── 6_报告导出.py              # HTML报告自动生成
│   ├── 7_干旱监测.py              # 多指数干旱 + 预测 + 沙漠化
│   ├── 8_冰冻圈分析.py            # NDSI雪盖 + 冰川 + 冻土
│   ├── 9_农业干旱.py              # CWSI + 土壤水分 + 灌溉
│   ├── 10_时序动画.py             # NDVI/水体/雪盖 GIF动画
│   ├── 11_生态评估.py             # PSR 生态安全评价
│   ├── 12_工作流.py               # AI智能查询 + 一键分析
│   ├── 13_土壤盐渍化.py            # SI/NDSI/BI + 5级盐渍化评估
│   ├── 14_LST.py                 # Landsat 热红外 LST 反演 + 热环境分级
│   ├── 15_指数计算器.py            # 预设指数 + 自定义波段运算
│   ├── 16_土地转移.py              # 双时相土地覆盖转移矩阵
│   ├── 17_蒸散发.py               # SEBAL 能量平衡蒸散发估算
│   ├── 18_监督分类.py              # 自定义样本训练模型 (RF/SVM/KNN/MLP)
│   ├── 19_矢量导出.py              # 分类结果矢量化 (GeoJSON/Shapefile/KML)
│   └── 20_数据下载中心.py           # 结果持久化 + 统一下载/打包/删除
├── utils/                        # 工具函数库 (27 模块)
│   ├── pc_data.py                # Planetary Computer STAC 数据获取
│   ├── indices.py                # NDVI/MNDWI/EVI/AWEIsh 指数计算
│   ├── drought.py                # SPI/SPEI/VCI/TCI/VHI/NDDI/TVDI/CDI
│   ├── forecast.py               # SARIMA/LSTM/Holt-Winters 预测
│   ├── desertification.py        # Albedo/TGSI/NDMI/DDI 沙漠化评估
│   ├── salinity.py               # SI/SI1/SI2/NDSI/BI 盐渍化评估
│   ├── lst.py                    # Landsat ST 地表温度反演 + 热环境分级
│   ├── spectral.py               # 波段运算求值器 + 预设指数库
│   ├── transition.py             # 土地覆盖转移矩阵计算
│   ├── evapotranspiration.py     # SEBAL 蒸散发估算 (能量平衡)
│   ├── supervised.py             # 监督分类训练 (sklearn 分类器)
│   ├── vector.py                 # 矢量导出 (栅格→GeoJSON/Shapefile/KML)
│   ├── results_store.py          # 结果持久化 (统一存储/检索/打包/删除)
│   ├── cryosphere.py             # NDSI/冰川/雪线/冻土
│   ├── agri_drought.py           # CWSI/SMI/MPDI/灌溉需求
│   ├── ecology.py                # PSR 生态安全模型
│   ├── animation.py              # GIF 时序动画合成
│   ├── llm.py                    # DeepSeek AI 智能查询
│   ├── ai_insight.py             # DeepSeek AI 智能解读引擎 (含规则降级)
│   ├── ai_engine.py              # geoai-py AI推理引擎
│   ├── landcover.py              # ESA/ESRI 土地覆盖
│   ├── trend.py                  # Sen+MK 趋势分析
│   ├── visualization.py          # 影像可视化
│   ├── export.py                 # GeoTIFF/CSV 导出
│   ├── onnx_engine.py            # ONNX Runtime 推理加速
│   ├── aoi.py                    # 自定义研究区 (AOI) 选择组件 + GeoJSON 解析
│   └── error_handler.py          # 统一错误处理
├── tests/                        # 单元测试
│   ├── test_core_modules.py      # 核心模块测试 (drought/desert/forecast...)
│   ├── test_salinity.py          # 盐渍化模块测试
│   ├── test_lst.py               # 地表温度模块测试
│   ├── test_spectral.py          # 波段运算/指数计算器测试
│   ├── test_transition.py        # 土地转移矩阵测试
│   ├── test_aoi.py               # 自定义 AOI 组件测试
│   ├── test_evapotranspiration.py # 蒸散发估算测试
│   ├── test_supervised.py        # 监督分类训练测试
│   ├── test_vector.py            # 矢量导出测试
│   └── test_results_store.py     # 结果持久化测试
├── config.py                     # 全局配置 + 研究区BBOX
├── models/                       # AI模型存储
├── .streamlit/
│   ├── config.toml               # 主题 + 服务器配置
│   └── secrets.toml.template     # 密钥模板
├── Dockerfile                    # Docker 容器化
├── requirements.txt              # Python 依赖完整清单
├── packages.txt                  # Streamlit Cloud 系统依赖
└── runtime.txt                   # Python 3.11
```

## 研究区域 (6 个预设 + 自定义 AOI)

### 预设研究区

| 研究区 | BBOX [lon_min, lat_min, lon_max, lat_max] | 推荐分析 |
|--------|------|------|
| 塔里木盆地 | [76, 37, 88, 42] | 🏜️ 沙漠化 + 🌿 绿洲植被 |
| 柴达木盆地 | [94, 36, 98, 39] | 💧 盐湖 + ❄️ 冰川 |
| 河西走廊 | [96, 38, 104, 42] | 🌾 绿洲农业 + 💧 水体 |
| 吐鲁番盆地 | [88, 42, 90, 43.5] | 🏜️ 极端干旱 + 🌡️ 高温 |
| 天山北坡 | [82, 43, 90, 45] | ❄️ 冰川融水 + 🌾 灌溉农业 |
| 准噶尔盆地 | [82, 44, 92, 48] | 🌿 荒漠草原 + 🏜️ 干旱 |

### 自定义 AOI (v1.10+)

所有分析页面支持**任意研究区**，不再局限于 6 个预设：

- 📍 **手动输入 bbox** — 直接输入经纬度范围
- 🗺️ **GeoJSON 上传** — 上传矢量边界文件，自动解析外接矩形 (支持 FeatureCollection / Feature / Geometry)
- 统一写入 session_state，跨页面共享

## 部署

### Streamlit Community Cloud (免费)
1. Push 代码到 GitHub
2. [share.streamlit.io](https://share.streamlit.io) → 连接仓库
3. 设置 `main` 分支, 主文件 `app.py`
4. Advanced Settings → Python 3.11 → Secrets 配置

### Docker
```bash
docker build -t geo-ai-app . && docker run -p 8501:8501 geo-ai-app
```

## 状态

- **版本**: v1.15 | **页面**: 20 | **工具模块**: 27
- **测试**: 156 用例全部通过 | **部署**: Streamlit Cloud ✅
- **Python**: 3.11 | **PyTorch**: 2.11.0+cpu

## License

MIT License
