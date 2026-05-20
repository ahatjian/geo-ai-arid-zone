# Geo AI 干旱区遥感智能分析平台 — Docker 部署
# 基于 Python 3.11-slim + GDAL + Streamlit

FROM python:3.11-slim-bookworm

LABEL maintainer="Geo AI Team"
LABEL description="西北干旱区遥感智能分析平台"

# ============================================
# 系统依赖
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # GDAL + Rasterio
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    # OpenCV
    libgl1-mesa-glx \
    libglib2.0-0 \
    # 构建工具
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# 环境变量
# ============================================
ENV GDAL_VERSION=3.8.5
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# ============================================
# 工作目录
# ============================================
WORKDIR /app

# ============================================
# Python 依赖
# ============================================
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================
# 应用代码
# ============================================
COPY . .

# ============================================
# 数据目录 (卷挂载点)
# ============================================
RUN mkdir -p /app/data /app/models /app/.cache /app/downloads

# ============================================
# 暴露端口
# ============================================
EXPOSE 8501

# ============================================
# 健康检查
# ============================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import streamlit; print('OK')" || exit 1

# ============================================
# 启动
# ============================================
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
