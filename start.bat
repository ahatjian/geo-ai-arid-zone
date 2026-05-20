@echo off
chcp 65001 >nul
echo ==========================================
echo  Geo AI 遥感分析平台 - 启动器
echo ==========================================
echo.

cd /d "%~dp0"

echo [1/2] 检查并安装新依赖 (xarray, rioxarray)...
pip install xarray rioxarray -i https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo [2/2] 启动 Streamlit 应用...
echo ==========================================
streamlit run app.py

echo.
pause
