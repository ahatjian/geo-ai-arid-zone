"""时序动画模块测试 — animation

三个公开函数都靠 matplotlib 渲染 + imageio/PIL 合成 GIF,
测试用极小尺寸 (8×8, 3 帧, 40 dpi) 保持快速, 只验证契约:
返回类型、GIF 魔数、输出路径分支、参数开关、退化输入。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


GIF_MAGIC = b"GIF8"  # GIF87a / GIF89a 均以此开头

# 小尺寸渲染参数, 避免每个用例耗时过长
FAST = dict(dpi=40, figsize=(2, 2))


def _stack(h=8, w=8, t=3, seed=0):
    return np.random.default_rng(seed).random((h, w, t))


def _dates(n):
    return [f"2024-{i + 1:02d}-01" for i in range(n)]


# ============================================================
# create_timeseries_animation
# ============================================================

class TestTimeseriesAnimation:
    def test_returns_gif_bytes(self):
        from utils.animation import create_timeseries_animation
        out = create_timeseries_animation(_stack(), _dates(3), **FAST)
        assert isinstance(out, bytes)
        assert out[:4] == GIF_MAGIC
        assert len(out) > 100

    def test_writes_file_and_returns_none(self, tmp_path):
        from utils.animation import create_timeseries_animation
        path = tmp_path / "anim.gif"
        out = create_timeseries_animation(_stack(), _dates(3),
                                          output_path=str(path), **FAST)
        assert out is None
        assert path.exists()
        assert path.read_bytes()[:4] == GIF_MAGIC

    def test_2d_input_raises_value_error(self):
        from utils.animation import create_timeseries_animation
        with pytest.raises(ValueError, match="3D"):
            create_timeseries_animation(np.zeros((8, 8)), _dates(1), **FAST)

    def test_1d_input_raises(self):
        from utils.animation import create_timeseries_animation
        with pytest.raises(ValueError):
            create_timeseries_animation(np.zeros(8), _dates(1), **FAST)

    def test_flags_disabled(self):
        """关掉日期/统计/色条仍应产出 GIF"""
        from utils.animation import create_timeseries_animation
        out = create_timeseries_animation(
            _stack(), _dates(3),
            add_colorbar=False, show_date=False, show_stat=False, **FAST,
        )
        assert out[:4] == GIF_MAGIC

    def test_explicit_vmin_vmax(self):
        from utils.animation import create_timeseries_animation
        out = create_timeseries_animation(_stack(), _dates(3),
                                          vmin=0.0, vmax=1.0, **FAST)
        assert out[:4] == GIF_MAGIC

    def test_single_frame(self):
        from utils.animation import create_timeseries_animation
        out = create_timeseries_animation(_stack(t=1), _dates(1), **FAST)
        assert out[:4] == GIF_MAGIC

    def test_dates_shorter_than_frames(self):
        """日期标签少于帧数时不应越界, 缺标签的帧照常渲染"""
        from utils.animation import create_timeseries_animation
        out = create_timeseries_animation(_stack(t=4), ["2024-01-01"], **FAST)
        assert out[:4] == GIF_MAGIC

    def test_constant_stack_does_not_crash(self):
        """全常量数组 → vmin == vmax, imshow 不应报错"""
        from utils.animation import create_timeseries_animation
        out = create_timeseries_animation(np.full((8, 8, 3), 0.4), _dates(3), **FAST)
        assert out[:4] == GIF_MAGIC

    def test_stack_with_nan(self):
        """含 NaN 的时相立方体应能渲染 (缺失像元常见于云掩膜后)"""
        from utils.animation import create_timeseries_animation
        stack = _stack()
        stack[2:5, 2:5, 1] = np.nan
        out = create_timeseries_animation(stack, _dates(3), **FAST)
        assert out[:4] == GIF_MAGIC

    def test_empty_stack_returns_none(self):
        """无时相数据 (T=0) 应返回 None, 而非在渲染/合成阶段崩溃"""
        from utils.animation import create_timeseries_animation
        out = create_timeseries_animation(np.zeros((8, 8, 0)), [], **FAST)
        assert out is None


# ============================================================
# create_multi_index_animation
# ============================================================

class TestMultiIndexAnimation:
    def test_empty_dict_returns_none(self):
        from utils.animation import create_multi_index_animation
        assert create_multi_index_animation({}, []) is None

    def test_single_index(self):
        """单指数走 axes 标量分支 (matplotlib 不返回列表)"""
        from utils.animation import create_multi_index_animation
        out = create_multi_index_animation({"NDVI": _stack()}, _dates(3), dpi=40)
        assert out[:4] == GIF_MAGIC

    def test_two_indices_use_default_configs(self):
        """NDVI/MNDWI 命中内置 cmap 配置"""
        from utils.animation import create_multi_index_animation
        out = create_multi_index_animation(
            {"NDVI": _stack(seed=1), "MNDWI": _stack(seed=2)},
            _dates(3), dpi=40,
        )
        assert out[:4] == GIF_MAGIC

    def test_unknown_index_falls_back_to_viridis(self):
        """未在内置配置中的指数名应回落默认 cmap 而非 KeyError"""
        from utils.animation import create_multi_index_animation
        out = create_multi_index_animation({"自定义指数": _stack()}, _dates(3), dpi=40)
        assert out[:4] == GIF_MAGIC

    def test_explicit_configs(self):
        from utils.animation import create_multi_index_animation
        out = create_multi_index_animation(
            {"A": _stack()}, _dates(3), dpi=40,
            configs={"A": {"cmap": "viridis", "vmin": 0, "vmax": 1, "label": "A"}},
        )
        assert out[:4] == GIF_MAGIC

    def test_empty_timestack_returns_none(self):
        """指数存在但时相维度为 0 时应返回 None"""
        from utils.animation import create_multi_index_animation
        out = create_multi_index_animation({"NDVI": np.zeros((8, 8, 0))}, [], dpi=40)
        assert out is None


# ============================================================
# create_trend_animation
# ============================================================

class TestTrendAnimation:
    def test_returns_gif_bytes(self):
        from utils.animation import create_trend_animation
        out = create_trend_animation(np.array([0.3, 0.35, 0.4]), _dates(3))
        assert isinstance(out, bytes)
        assert out[:4] == GIF_MAGIC

    def test_highlight_last(self):
        from utils.animation import create_trend_animation
        out = create_trend_animation(np.array([0.3, 0.35, 0.4, 0.45]),
                                     _dates(4), highlight_last=2)
        assert out[:4] == GIF_MAGIC

    def test_single_point(self):
        from utils.animation import create_trend_animation
        out = create_trend_animation(np.array([0.3]), _dates(1))
        assert out[:4] == GIF_MAGIC

    def test_constant_series(self):
        """全常量序列 → ylim 上下界相同, 不应报错"""
        from utils.animation import create_trend_animation
        out = create_trend_animation(np.full(3, 0.4), _dates(3))
        assert out[:4] == GIF_MAGIC

    def test_dates_shorter_than_series(self):
        from utils.animation import create_trend_animation
        out = create_trend_animation(np.array([0.3, 0.4, 0.5]), ["2024-01-01"])
        assert out[:4] == GIF_MAGIC

    def test_accepts_list_input(self):
        from utils.animation import create_trend_animation
        out = create_trend_animation([0.3, 0.35, 0.4], _dates(3))
        assert out[:4] == GIF_MAGIC

    def test_empty_series_returns_none(self):
        """空时序应返回 None, 而非在合成阶段索引空列表"""
        from utils.animation import create_trend_animation
        out = create_trend_animation(np.array([]), [])
        assert out is None
