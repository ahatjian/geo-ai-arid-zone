@echo off
chcp 65001 >nul
echo ==========================================
echo  在 base 环境安装所有依赖
echo ==========================================
echo.

cd /d "%~dp0"

echo [1/3] 安装 Streamlit...
pip install streamlit -i https://pypi.tuna.tsinghua.edu.cn/simple

echo [2/3] 安装 leafmap...
pip install leafmap -i https://pypi.tuna.tsinghua.edu.cn/simple

echo [3/3] 安装 Planetary Computer 工具...
pip install pystac-client planetary-computer rasterio -i https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo ==========================================
echo  安装完成，正在启动应用...
echo ==========================================
streamlit run app.py

pause
