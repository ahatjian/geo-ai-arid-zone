@echo off
chcp 65001 >nul
echo ==========================================
echo  Geo AI 遥感分析平台 - 一键启动器
echo ==========================================
echo.

cd /d "%~dp0"

REM 激活 conda 环境 (如存在)
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat" geo-ai 2>nul
    if errorlevel 1 (
        echo [提示] 未找到 geo-ai 环境, 使用当前 Python
    ) else (
        echo [OK] 已激活 geo-ai 环境
    )
)

echo.
echo [1/3] 环境健康检查...
python check_env.py
if errorlevel 1 (
    echo.
    echo [错误] 环境检查未通过, 请先安装依赖:
    echo     pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo [2/3] 检查 DeepSeek API Key...
python -c "import os;from pathlib import Path;p=Path('.streamlit/secrets.toml');ok='DEEPSEEK_API_KEY' in p.read_text(encoding='utf-8') if p.exists() else bool(os.environ.get('DEEPSEEK_API_KEY'));print('   [OK] DeepSeek AI 已配置' if ok else '   [提示] 未配置 DeepSeek Key — 将使用模板匹配模式')"

echo.
echo [3/3] 启动 Streamlit 应用...
echo ==========================================
echo  浏览器将自动打开: http://localhost:8501
echo  关闭此窗口即停止应用
echo ==========================================
echo.
streamlit run app.py

pause
