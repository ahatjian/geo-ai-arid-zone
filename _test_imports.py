"""全量导入验证"""
import sys, traceback, importlib
errors = []

modules = [
    'config',
    'utils.pc_data',
    'utils.indices',
    'utils.trend',
    'utils.ai_engine',
    'utils.export',
    'utils.landcover',
    'utils.visualization',
    'utils.error_handler',
]

for mod in modules:
    try:
        importlib.import_module(mod)
        print(f'✅ {mod}')
    except Exception as e:
        print(f'❌ {mod}: {e}')
        errors.append(mod)
        traceback.print_exc()

if errors:
    print(f'\n❌ {len(errors)} modules FAILED: {errors}')
    sys.exit(1)
else:
    print(f'\n✅ 全部 {len(modules)} 个模块导入通过')
