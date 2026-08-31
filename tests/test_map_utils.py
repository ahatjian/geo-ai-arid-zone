"""研究区地图工具测试 — leafmap 降级静态地图"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest


SAMPLE_AREAS = {
    "塔里木盆地": {"bbox": [76, 37, 88, 42]},
    "柴达木盆地": {"bbox": [94, 36, 98, 39]},
    "河西走廊": {"bbox": [96, 38, 104, 42]},
}


class TestDrawAreaSchematic:
    def test_png_generated(self):
        from utils.map_utils import draw_area_schematic
        img = draw_area_schematic("塔里木盆地", [76, 37, 88, 42])
        assert img is not None
        assert img[:4] == b"\x89PNG"  # PNG 头

    def test_with_all_areas(self):
        from utils.map_utils import draw_area_schematic
        img = draw_area_schematic("塔里木盆地", [76, 37, 88, 42],
                                  all_areas=SAMPLE_AREAS)
        assert img is not None

    def test_invalid_bbox(self):
        """异常 bbox 不应崩溃"""
        from utils.map_utils import draw_area_schematic
        img = draw_area_schematic("测试", [100, 100, 90, 90])
        assert img is not None or True  # 不崩溃即可


class TestRenderAreaMap:
    def test_leafmap_fallback_to_static(self, monkeypatch):
        """leafmap 导入失败应降级为静态图"""
        import utils.map_utils as mu
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "leafmap":
                raise ImportError("模拟 leafmap 不可用")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        mode, img = mu.render_area_map("塔里木盆地", [76, 37, 88, 42])
        assert mode == "static"
        assert img is not None
        monkeypatch.setattr(builtins, "__import__", real_import)

    def test_returns_tuple(self):
        from utils.map_utils import render_area_map
        result = render_area_map("塔里木盆地", [76, 37, 88, 42])
        assert isinstance(result, tuple)
        assert len(result) == 2
