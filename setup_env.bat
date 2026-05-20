@echo off
chcp 65001 >nul
echo ==========================================
echo  安装 Web-Geo-AI 项目依赖
echo ==========================================
echo.

REM 激活conda环境
call conda activate geo-ai
if errorlevel 1 (
    echo [错误] 无法激活 geo-ai 环境，请先创建：
    echo   conda create -n geo-ai python=3.10 -y
    pause
    exit /b 1
)

echo [1/4] 正在安装 Streamlit...
pip install streamlit -i https://pypi.tuna.tsinghua.edu.cn/simple

echo [2/4] 正在安装 leafmap...
pip install leafmap -i https://pypi.tuna.tsinghua.edu.cn/simple

echo [3/4] 正在安装 Planetary Computer 相关依赖...
pip install pystac-client planetary-computer rasterio -i https://pypi.tuna.tsinghua.edu.cn/simple

echo [4/4] 正在安装 AI/ML 依赖...
pip install torch torchvision segmentation-models-pytorch -i https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo ==========================================
echo  安装完成！
echo ==========================================
echo.
echo 现在可以运行：streamlit run app.py
echo.
pause
