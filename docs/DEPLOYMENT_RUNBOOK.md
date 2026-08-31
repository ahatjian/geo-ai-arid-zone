# 西北干旱区 Geo AI 平台 — 精简部署 Runbook

> 版本: v1.17 | 最后更新: 2026-08-30 | 维护: Geo AI Team

---

## 1. 何时使用此 Runbook

| 场景 | 参考节 |
|------|--------|
| 刚部署完需要验证 | §3 健康检查 |
| 新版本上线后异常 | §4 回滚方案 |
| 应用无法启动 | §5.1 启动失败 |
| 页面加载缓慢/白屏 | §5.2 性能与超时 |
| 数据搜索不返回 | §5.3 STAC 数据源 |
| 模型推理报错 | §5.4 ONNX / AI 模型 |
| Streamlit Cloud 部署失败 | §5.5 Cloud 部署 |

---

## 2. 前置条件与所需权限

| 环境 | 所需权限 |
|------|----------|
| **本地** | Python 3.11, Git, 可访问 Planetary Computer API |
| **Docker** | Docker Engine, 端口 8501 可用 |
| **Streamlit Cloud** | GitHub 仓库 Admin 权限, Streamlit Cloud 账号 |

---

## 3. 健康检查

### 3.1 基础环境检查

```bash
# 1. Python 版本
python --version
# 预期: Python 3.11.x

# 2. 关键依赖导入
python -c "
import streamlit; print('Streamlit:', streamlit.__version__)
import torch; print('PyTorch:', torch.__version__)
import rasterio; print('Rasterio:', rasterio.__version__)
import onnxruntime; print('ONNX Runtime:', onnxruntime.__version__)
print('All imports OK')
"

# 3. GDAL 系统库 (Linux)
gdalinfo --version
# 预期: GDAL 3.x

# 4. 目录完整性
python -c "
import os
required = ['pages','utils','models','data','.streamlit','docs']
for d in required:
    exists = os.path.isdir(d)
    print(f'  {d}: {\"OK\" if exists else \"MISSING\"}')"
```

### 3.2 应用启动检查

```bash
# 本地启动 (前台, 观察启动日志)
streamlit run app.py --server.port=8501

# 预期日志:
#   You can now view your Streamlit app in your browser.
#   Local URL: http://localhost:8501
```

**关键验证点** (浏览器访问 http://localhost:8501):
- [ ] 首页显示 6 个研究区卡片
- [ ] 侧边栏显示 12 个功能页面
- [ ] 选择一个研究区 → 页面正常跳转
- [ ] 数据浏览页 → STAC 搜索返回影像
- [ ] 无 "ModuleNotFoundError" 或 "ImportError"

### 3.3 Docker 健康检查 (Dockerfile 内置)

```bash
# Dockerfile 已配置 HEALTHCHECK:
#   HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
#       CMD python -c "import streamlit; print('OK')" || exit 1

# 手动检查容器健康状态
docker ps --filter "name=geo-ai"
docker inspect --format='{{.State.Health.Status}}' <container_id>
```

### 3.4 Streamlit Cloud 部署验证

1. 浏览器访问 `https://<your-app>.streamlit.app`
2. 查看右下角 "Manage app" → 检查日志
3. 确认 `packages.txt` 中的系统依赖全部安装成功 (无 `apt-get` 错误)

---

## 4. 回滚方案

### 4.1 快速回滚 (Git)

```bash
# 1. 查看最近的部署标签
git tag --sort=-creatordate | head -5

# 2. 回滚到上一个稳定版本 (使用 tag)
git checkout v1.6

# 3. 如果使用分支部署
git revert <坏提交的hash> --no-edit

# 4. 推送回滚
git push origin main

# 5. Streamlit Cloud 会自动检测并重新部署
```

### 4.2 Docker 回滚

```bash
# 1. 停止当前容器
docker stop geo-ai-app && docker rm geo-ai-app

# 2. 使用上一个稳定镜像
docker run -d --name geo-ai-app -p 8501:8501 geo-ai-app:v1.6

# 3. 验证
curl -s http://localhost:8501/_stcore/health
```

### 4.3 回滚决策矩阵

| 问题类型 | 回滚方式 | 预计恢复时间 |
|----------|----------|-------------|
| 依赖版本不兼容 | `git revert` + 推送 | 2-5 分钟 |
| 新功能导致报错 | `git revert` 对应提交 | 2-5 分钟 |
| 模型文件损坏 | 恢复 models/ 上一版本 | 1 分钟 |
| 配置文件错误 | 回滚 config.py | 1 分钟 |

---

## 5. 故障排查指南

### 5.1 应用启动失败

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| `ModuleNotFoundError: No module named 'streamlit'` | 依赖未安装 | `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'rasterio'` | GDAL 系统库缺失 | `apt-get install gdal-bin libgdal-dev` (Linux) |
| `ImportError: libGL.so.1` | OpenCV 依赖缺失 | `apt-get install libgl1-mesa-glx libglib2.0-0` |
| `OSError: cannot load library 'gdal302'` | GDAL 版本不匹配 | 重装 rasterio: `pip uninstall rasterio -y && pip install rasterio` |
| `PermissionError` on port 8501 | 端口被占用 | `lsof -i :8501` 查看占用, 换端口 `--server.port=8502` |

### 5.2 页面加载缓慢 / 白屏

| 症状 | 诊断命令 | 解决方案 |
|------|----------|----------|
| 首次加载 >2 分钟 | 查看日志中的模型下载 | 预下载模型到 `models/` 目录 |
| AI 推理超时 (CPU) | 检查 ONNX 是否启用 | 确认 `config.py` 中 `ONNX_CONFIG["enabled"] = True` |
| 大数据集加载缓慢 | 检查 `CACHE_CONFIG` | 增大 `ttl_medium` / `ttl_long` 减少重复计算 |
| 浏览器白屏 | 查看 Streamlit 日志 | 检查是否有未捕获异常, 增加 `maxMessageSize` |

```bash
# 检查 Streamlit 日志
# 本地: 终端输出
# Docker: docker logs geo-ai-app --tail 100
# Cloud: https://share.streamlit.io → 你的 App → "Manage app" → Logs

# 检查 ONNX 是否正常
python -c "
import onnxruntime as ort
print('Available providers:', ort.get_available_providers())
print('Expected: [CPUExecutionProvider]')
"
```

### 5.3 STAC 数据源问题

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| 搜索无结果 | 研究区 BBOX 无效 | 检查 `config.py` → `STUDY_AREAS` 的 bbox 格式 |
| `ConnectionError` / `Timeout` | Planetary Computer 不可达 | 检查网络, 尝试 `curl https://planetarycomputer.microsoft.com/api/stac/v1` |
| `KeyError: 'assets'` | STAC Item 结构变化 | 检查 `utils/pc_data.py` 中的 band 名称 |
| 下载极慢 | 影像范围过大 | 缩小 BBOX 或降低时间跨度 |

```bash
# 快速测试 Planetary Computer 连通性
python -c "
from pystac_client import Client
cat = Client.open('https://planetarycomputer.microsoft.com/api/stac/v1')
print('STAC API 连接成功')
search = cat.search(collections=['sentinel-2-l2a'], limit=1)
print(f'找到 {search.matched()} 个 Sentinel-2 影像')
"
```

### 5.4 AI 模型 / ONNX 推理错误

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| `FileNotFoundError: models/water_seg_*.onnx` | ONNX 模型未下载 | 运行 `python -c "from geoai import segment_water; print('OK')"` |
| `ONNX Runtime Error` | ONNX 版本不兼容 | 升级 `pip install onnxruntime>=1.18.0 --upgrade` |
| PyTorch 推理缓慢 | ONNX 未启用 | 确认 `ONNX_CONFIG["enabled"] = True` |
| `OutOfMemoryError` (CPU) | 影像切片过大 | 减小 `ONNX_CONFIG["window_size"]` 或 `batch_size` |
| `geoai-py` 导入失败 | 可选依赖未安装 | 注释掉相关代码或安装 `pip install geoai-py` |

```bash
# 手动验证 ONNX 模型
python -c "
import os
models_dir = 'models'
for f in os.listdir(models_dir):
    if f.endswith('.onnx'):
        print(f'ONNX model found: {os.path.join(models_dir, f)} ({os.path.getsize(os.path.join(models_dir, f))/1e6:.1f} MB)')
"
```

### 5.5 Streamlit Cloud 部署故障

| 症状 | 诊断 | 解决方案 |
|------|------|----------|
| Deploy 一直 "Loading" | 查看部署日志 | 检查 `runtime.txt` 是否为 `3.11` |
| `apt-get` 报错 | 系统依赖缺失 | 检查 `packages.txt` 中包名是否正确 |
| `ModuleNotFoundError` on Cloud | requirements 不完整 | 本地 `pip freeze` 对比 `requirements.txt` |
| 超出内存限制 (1GB) | 数据/模型过大 | 使用 `@st.cache_data` 减少内存, 优化切片大小 |
| Reboot loop | 启动时异常 | 检查 `app.py` 顶层代码, 用 try/except 包裹 |

```bash
# 本地模拟 Cloud 环境
pip install -r requirements.txt --no-cache-dir
streamlit run app.py --server.headless=true --browser.gatherUsageStats=false
```

---

## 6. 关键配置参考

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `STREAMLIT_SERVER_PORT` | 8501 | 服务端口 |
| `STREAMLIT_SERVER_ADDRESS` | 0.0.0.0 | 绑定地址 |
| `PYTHONUNBUFFERED` | 1 | 日志实时输出 |
| `DEEPSEEK_API_KEY` | (无) | DeepSeek AI Key, 缺失时自动降级为模板匹配 |
| `DEEPSEEK_MODEL` | deepseek-chat | DeepSeek 模型名 (可选) |
| `DEEPSEEK_TEMPERATURE` | 0.3 | LLM 采样温度 (可选) |

### 端口 (防火墙规则)

| 端口 | 用途 |
|------|------|
| 8501 | Streamlit Web UI |
| 443 | Planetary Computer STAC API (出站) |

### 目录权限

```bash
# 确保以下目录可写
chmod -R 755 data/ models/ .cache/ downloads/
```

---

## 7. 升级路径 (Escalation)

| 级别 | 条件 | 联系 |
|------|------|------|
| L1 - 自助 | 文档可解决的常见问题 | 本 Runbook + README.md |
| L2 - 团队 | 需要代码修改, 非紧急 | Geo AI Team (GitHub Issues) |
| L3 - 紧急 | 生产环境不可用, 回滚无效 | 工程总监 + 值班 SRE |

---

## 8. 附录: 一键健康检查脚本

```bash
#!/bin/bash
# 保存为 scripts/health_check.sh
echo "=== Geo AI 平台健康检查 ==="
echo "时间: $(date)"
echo ""

# Python 版本
echo "[1/6] Python 版本:"
python --version

# 依赖导入
echo "[2/6] 关键依赖导入:"
python -c "import streamlit,torch,rasterio,onnxruntime; print('OK')" && echo "PASS" || echo "FAIL"

# 目录
echo "[3/6] 目录完整性:"
for d in pages utils models data .streamlit; do
    [ -d "$d" ] && echo "  $d: OK" || echo "  $d: MISSING"
done

# ONNX 模型
echo "[4/6] ONNX 模型文件:"
ls -lh models/*.onnx 2>/dev/null || echo "  (无 .onnx 文件, 将自动下载)"

# 端口
echo "[5/6] 端口 8501:"
lsof -i :8501 2>/dev/null && echo "  (已被占用)" || echo "  (可用)"

# STAC 连接
echo "[6/6] STAC API:"
python -c "from pystac_client import Client; Client.open('https://planetarycomputer.microsoft.com/api/stac/v1'); print('PASS')" 2>/dev/null || echo "FAIL"

echo ""
echo "=== 检查完成 ==="
```

---

**维护日志**:
- 2026-04-21: 初始版本 (v1.7), 覆盖 12 页应用全部模块
- 2026-08-30: 升级至 v1.17, 覆盖 20 页应用全部模块, DeepSeek AI 智能查询与解读
