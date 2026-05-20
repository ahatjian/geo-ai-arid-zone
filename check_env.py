"""
环境诊断脚本 - 检查Python环境配置
"""
import sys
import subprocess

print("=" * 50)
print("Python环境诊断")
print("=" * 50)
print(f"\nPython路径: {sys.executable}")
print(f"Python版本: {sys.version}")
print(f"\n当前环境: {sys.prefix}")

# 检查streamlit
try:
    import streamlit
    print(f"\n✅ Streamlit 已安装")
    print(f"   版本: {streamlit.__version__}")
    print(f"   路径: {streamlit.__file__}")
except ImportError:
    print(f"\n❌ Streamlit 未安装")

# 检查leafmap
try:
    import leafmap
    print(f"\n✅ leafmap 已安装")
    print(f"   版本: {leafmap.__version__}")
except ImportError:
    print(f"\n❌ leafmap 未安装")

# 检查conda环境
print(f"\n{'=' * 50}")
print("建议解决方案:")
print(f"{'=' * 50}")

if "geo-ai" not in sys.prefix:
    print("\n⚠️  当前不在 geo-ai 环境中！")
    print("\n请执行以下命令:")
    print("   conda activate geo-ai")
    print("   python check_env.py")
else:
    print("\n✅ 已在 geo-ai 环境中")
    
print("\n或者直接使用 base 环境运行:")
print(f"   {sys.executable.split('python.exe')[0]}Scripts\streamlit.exe run app.py")
