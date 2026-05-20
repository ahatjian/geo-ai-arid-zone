@echo off
chcp 65001 >nul
echo ==========================================
echo  启动 Web-Geo-AI 应用
echo ==========================================
echo.

REM 激活conda环境并运行
call conda activate geo-ai && streamlit run app.py

pause
