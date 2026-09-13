"""ONNX 推理引擎测试 — onnx / onnxruntime

onnx 包在本项目中是**非必需依赖**（模型由用户上传，见 README 与第 2/4 页），
因此测试分两层:

* 算法层 — 用 InferenceSession 替身, 验证滑窗切分、边缘 padding、
  重叠归一化、激活函数分支、进度回调; 不依赖任何 .onnx 文件
* 降级层 — 无 onnx 包 / 无模型文件 / 波段越界时的错误处理
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


# ============================================================
# InferenceSession 替身
# ============================================================

class _FakeTensor:
    def __init__(self, name):
        self.name = name


class _FakeSession:
    """最小 InferenceSession 替身, 记录每次 run 实际收到的张量形状。

    真实 ONNX Runtime 无法在没有 .onnx 文件时构造, 而滑窗拼接逻辑
    (本模块最易出错的部分) 完全由喂进 session 的张量形状决定,
    因此记录 feed 形状即可完整验证。
    """

    def __init__(self, out_channels: int = 2, constant: float = 0.0):
        self.out_channels = out_channels
        self.constant = constant
        self.feed_shapes = []
        self.run_count = 0

    def get_inputs(self):
        return [_FakeTensor("input")]

    def get_outputs(self):
        return [_FakeTensor("output")]

    def run(self, output_names, feed):
        self.run_count += 1
        x = feed["input"]
        self.feed_shapes.append(tuple(x.shape))
        b, _, h, w = x.shape
        logits = np.full((b, self.out_channels, h, w), self.constant, dtype=np.float32)
        # 让最后一个通道略高, 便于断言 argmax 结果
        logits[:, -1] = self.constant + 1.0
        return [logits]


@pytest.fixture
def fake_session(monkeypatch):
    """把 get_ort_session 换成替身, 并返回替身供断言"""
    def _install(out_channels=2, constant=0.0):
        from utils import onnx_engine
        session = _FakeSession(out_channels=out_channels, constant=constant)
        monkeypatch.setattr(onnx_engine, "get_ort_session", lambda *a, **kw: session)
        return session
    return _install


# ============================================================
# _softmax
# ============================================================

class TestSoftmax:
    def test_sums_to_one(self):
        from utils.onnx_engine import _softmax
        x = np.random.default_rng(0).normal(size=(4, 8, 8)).astype(np.float32)
        probs = _softmax(x, axis=0)
        assert np.allclose(probs.sum(axis=0), 1.0, atol=1e-5)

    def test_numerically_stable_on_large_input(self):
        """大 logits 不应溢出为 nan/inf (朴素 exp 会)"""
        from utils.onnx_engine import _softmax
        x = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
        probs = _softmax(x, axis=0)
        assert np.isfinite(probs).all()
        assert np.isclose(probs.sum(), 1.0, atol=1e-5)

    def test_very_negative_input(self):
        from utils.onnx_engine import _softmax
        x = np.array([-1000.0, -1001.0, -1002.0], dtype=np.float32)
        probs = _softmax(x, axis=0)
        assert np.isfinite(probs).all()

    def test_axis_argument(self):
        from utils.onnx_engine import _softmax
        x = np.zeros((3, 5), dtype=np.float32)
        assert np.allclose(_softmax(x, axis=0).sum(axis=0), 1.0)
        assert np.allclose(_softmax(x, axis=1).sum(axis=1), 1.0)


# ============================================================
# check_onnx_available
# ============================================================

class TestCheckOnnxAvailable:
    def test_returns_expected_schema(self):
        from utils.onnx_engine import check_onnx_available
        info = check_onnx_available()
        for key in ("onnx_available", "providers", "version",
                    "cuda_available", "recommended_provider"):
            assert key in info

    def test_onnxruntime_installed(self):
        """本环境已装 onnxruntime, 应报告可用"""
        from utils.onnx_engine import check_onnx_available
        info = check_onnx_available()
        assert info["onnx_available"] is True
        assert isinstance(info["providers"], list)
        assert info["recommended_provider"] in info["providers"]

    def test_missing_onnxruntime_degrades(self, monkeypatch):
        """onnxruntime 缺失时应返回不可用而非抛异常"""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "onnxruntime":
                raise ImportError("simulated: no onnxruntime")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        from utils.onnx_engine import check_onnx_available
        info = check_onnx_available()
        assert info["onnx_available"] is False
        assert info["providers"] == []


# ============================================================
# _ensure_chw — 维度判别
# ============================================================

class TestEnsureChw:
    def test_already_chw_untouched(self):
        from utils.onnx_engine import _ensure_chw
        arr = np.zeros((6, 64, 64), dtype=np.float32)
        assert _ensure_chw(arr).shape == (6, 64, 64)

    def test_hwc_transposed(self):
        from utils.onnx_engine import _ensure_chw
        arr = np.zeros((64, 64, 6), dtype=np.float32)
        assert _ensure_chw(arr).shape == (6, 64, 64)

    def test_transpose_actually_moves_data(self):
        """转置要真搬数据, 不能只改 shape"""
        from utils.onnx_engine import _ensure_chw
        arr = np.arange(20 * 20 * 6, dtype=np.float32).reshape(20, 20, 6)
        out = _ensure_chw(arr)
        assert out.shape == (6, 20, 20)
        assert out[0, 0, 0] == arr[0, 0, 0]
        assert out[1, 0, 0] == arr[0, 0, 1]
        assert out[5, 7, 9] == arr[7, 9, 5]

    def test_2d_passthrough(self):
        from utils.onnx_engine import _ensure_chw
        arr = np.zeros((16, 16), dtype=np.float32)
        assert _ensure_chw(arr).shape == (16, 16)

    def test_first_dim_equal_max_bands_is_chw(self):
        """首维恰好等于上限 (13 波段) 时应判为 (C, H, W)"""
        from utils.onnx_engine import _ensure_chw, MAX_BANDS
        arr = np.zeros((MAX_BANDS, 100, 100), dtype=np.float32)
        assert _ensure_chw(arr).shape == (MAX_BANDS, 100, 100)

    def test_last_dim_equal_max_bands_is_hwc(self):
        from utils.onnx_engine import _ensure_chw, MAX_BANDS
        arr = np.zeros((100, 100, MAX_BANDS), dtype=np.float32)
        assert _ensure_chw(arr).shape == (MAX_BANDS, 100, 100)

    def test_ambiguous_shape_warns_and_keeps_chw(self):
        """首尾维度都超过波段上限时无法判别, 应告警并按 (C, H, W) 处理"""
        from utils.onnx_engine import _ensure_chw
        arr = np.zeros((50, 200, 200), dtype=np.float32)
        with pytest.warns(UserWarning, match="无法判断"):
            out = _ensure_chw(arr)
        assert out.shape == (50, 200, 200)

    @pytest.mark.parametrize("bands", [4, 6, 10, 13])
    def test_sentinel2_band_counts_round_trip(self, bands):
        """平台实际用到的波段数, 两种布局都应归一到 (C, H, W)"""
        from utils.onnx_engine import _ensure_chw
        chw = np.zeros((bands, 128, 128), dtype=np.float32)
        hwc = np.zeros((128, 128, bands), dtype=np.float32)
        assert _ensure_chw(chw).shape == (bands, 128, 128)
        assert _ensure_chw(hwc).shape == (bands, 128, 128)


# ============================================================
# run_onnx_inference — 输入布局
# ============================================================

class TestRunOnnxInferenceInputLayout:
    """喂给 session 的张量必须是 (1, C, H, W)。

    生产调用点全部传 np.stack(bands, axis=0) 即 (C, H, W):
    * utils/onnx_engine.py:520
    * pages/4_AI分类.py:665
    """

    def test_chw_input_preserved(self, fake_session):
        from utils.onnx_engine import run_onnx_inference
        session = fake_session(out_channels=6)
        arr = np.random.default_rng(0).random((6, 64, 64)).astype(np.float32)

        run_onnx_inference("m.onnx", arr, window_size=64, overlap=0,
                           num_classes=6, activation="softmax")

        assert session.feed_shapes == [(1, 6, 64, 64)]

    def test_hwc_input_transposed(self, fake_session):
        from utils.onnx_engine import run_onnx_inference
        session = fake_session(out_channels=6)
        arr = np.random.default_rng(0).random((64, 64, 6)).astype(np.float32)

        run_onnx_inference("m.onnx", arr, window_size=64, overlap=0,
                           num_classes=6, activation="softmax")

        assert session.feed_shapes == [(1, 6, 64, 64)]

    def test_chw_with_many_bands(self, fake_session):
        """10 波段 (Sentinel-2 全波段) 不应被误判为 (H, W, C)"""
        from utils.onnx_engine import run_onnx_inference
        session = fake_session(out_channels=10)
        arr = np.random.default_rng(0).random((10, 96, 96)).astype(np.float32)

        run_onnx_inference("m.onnx", arr, window_size=96, overlap=0,
                           num_classes=10, activation="softmax")

        assert session.feed_shapes == [(1, 10, 96, 96)]

    def test_output_shape_matches_input(self, fake_session):
        """输出必须是 (H, W), 与输入空间尺寸一致"""
        from utils.onnx_engine import run_onnx_inference
        fake_session(out_channels=2)
        arr = np.random.default_rng(0).random((4, 50, 70)).astype(np.float32)

        result = run_onnx_inference("m.onnx", arr, window_size=50, overlap=0,
                                    num_classes=2, activation="softmax")

        assert result["output"].shape == (50, 70)
        assert result["prob_map"].shape == (50, 70, 2)


# ============================================================
# run_onnx_inference — 滑窗与边缘
# ============================================================

class TestRunOnnxInferenceWindows:
    def test_single_window(self, fake_session):
        from utils.onnx_engine import run_onnx_inference
        session = fake_session(out_channels=2)
        arr = np.zeros((3, 32, 32), dtype=np.float32)

        result = run_onnx_inference("m.onnx", arr, window_size=32, overlap=0,
                                    num_classes=2, activation="softmax")

        assert result["num_windows"] == 1
        assert session.run_count == 1

    def test_multiple_windows_count(self, fake_session):
        from utils.onnx_engine import run_onnx_inference
        session = fake_session(out_channels=2)
        arr = np.zeros((3, 128, 128), dtype=np.float32)

        result = run_onnx_inference("m.onnx", arr, window_size=64, overlap=0,
                                    num_classes=2, activation="softmax")

        # 128/64 → 2×2 窗口
        assert result["num_windows"] == 4
        assert session.run_count == 4

    def test_edge_window_padded_to_window_size(self, fake_session):
        """非整除尺寸的末窗需 reflect-pad 到 window_size"""
        from utils.onnx_engine import run_onnx_inference
        session = fake_session(out_channels=2)
        arr = np.zeros((3, 100, 100), dtype=np.float32)

        run_onnx_inference("m.onnx", arr, window_size=64, overlap=0,
                           num_classes=2, activation="softmax")

        # 100 = 64 + 36 → 末窗不足, 补到 64
        assert all(shape == (1, 3, 64, 64) for shape in session.feed_shapes)

    def test_overlap_weights_are_averaged(self, fake_session):
        """重叠区域应取平均, 边界值不应因累加而过冲"""
        from utils.onnx_engine import run_onnx_inference
        fake_session(out_channels=2, constant=0.0)
        arr = np.zeros((3, 64, 64), dtype=np.float32)

        result = run_onnx_inference("m.onnx", arr, window_size=32, overlap=16,
                                    num_classes=2, activation="softmax")

        probs = result["prob_map"]
        assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-4)
        # 第 1 类在所有窗口都更高 → 全图应判为 1
        assert (result["output"] == 1).all()

    def test_small_image_smaller_than_window(self, fake_session):
        """图像小于窗口时应走单窗 + 全幅 padding"""
        from utils.onnx_engine import run_onnx_inference
        session = fake_session(out_channels=2)
        arr = np.zeros((3, 20, 20), dtype=np.float32)

        result = run_onnx_inference("m.onnx", arr, window_size=64, overlap=0,
                                    num_classes=2, activation="softmax")

        assert result["output"].shape == (20, 20)
        assert session.feed_shapes == [(1, 3, 64, 64)]


# ============================================================
# run_onnx_inference — 激活函数
# ============================================================

class TestRunOnnxInferenceActivation:
    def test_softmax_multiclass(self, fake_session):
        from utils.onnx_engine import run_onnx_inference
        fake_session(out_channels=5)
        arr = np.zeros((4, 32, 32), dtype=np.float32)

        result = run_onnx_inference("m.onnx", arr, window_size=32, overlap=0,
                                    num_classes=5, activation="softmax")

        assert np.allclose(result["prob_map"].sum(axis=-1), 1.0, atol=1e-5)

    def test_sigmoid_binary_single_channel(self, fake_session):
        """二分类模型通常输出 1 通道, 应扩展为 2 类概率"""
        from utils.onnx_engine import run_onnx_inference
        fake_session(out_channels=1)
        arr = np.zeros((4, 32, 32), dtype=np.float32)

        result = run_onnx_inference("m.onnx", arr, window_size=32, overlap=0,
                                    num_classes=2, activation="sigmoid")

        assert result["prob_map"].shape == (32, 32, 2)
        assert np.allclose(result["prob_map"].sum(axis=-1), 1.0, atol=1e-5)

    def test_none_activation_passthrough(self, fake_session):
        from utils.onnx_engine import run_onnx_inference
        fake_session(out_channels=3, constant=0.5)
        arr = np.zeros((2, 24, 24), dtype=np.float32)

        result = run_onnx_inference("m.onnx", arr, window_size=24, overlap=0,
                                    num_classes=3, activation="none")

        # 未激活 → 概率图直接是原始 logits: 峰值 1.5 不应被压缩到 0-1
        probs = result["prob_map"]
        assert np.isclose(probs[..., 0].mean(), 0.5, atol=1e-5)
        assert np.isclose(probs[..., -1].mean(), 1.5, atol=1e-5)
        assert probs.max() > 1.0

    def test_output_dtype_is_integer_labels(self, fake_session):
        from utils.onnx_engine import run_onnx_inference
        fake_session(out_channels=2)
        arr = np.zeros((3, 32, 32), dtype=np.float32)

        result = run_onnx_inference("m.onnx", arr, window_size=32, overlap=0,
                                    num_classes=2, activation="softmax")

        assert np.issubdtype(result["output"].dtype, np.integer)


# ============================================================
# run_onnx_inference — 进度回调与返回值
# ============================================================

class TestRunOnnxInferenceCallbacks:
    def test_progress_callback_counts_up(self, fake_session):
        from utils.onnx_engine import run_onnx_inference
        fake_session(out_channels=2)
        calls = []

        run_onnx_inference("m.onnx", np.zeros((3, 128, 128), dtype=np.float32),
                           window_size=64, overlap=0, num_classes=2,
                           activation="softmax",
                           progress_callback=lambda done, total: calls.append((done, total)))

        assert calls == [(1, 4), (2, 4), (3, 4), (4, 4)]

    def test_result_contains_timing_and_window_count(self, fake_session):
        from utils.onnx_engine import run_onnx_inference
        fake_session(out_channels=2)

        result = run_onnx_inference("m.onnx", np.zeros((3, 32, 32), dtype=np.float32),
                                    window_size=32, overlap=0, num_classes=2,
                                    activation="softmax")

        assert "inference_time_s" in result
        assert result["inference_time_s"] >= 0
        assert result["num_windows"] == 1

    def test_provider_forwarded_to_session(self, monkeypatch):
        """provider 参数应透传给 get_ort_session"""
        from utils import onnx_engine
        captured = {}

        def fake_get_session(path, provider=None, session_options=None):
            captured["provider"] = provider
            captured["path"] = path
            return _FakeSession()

        monkeypatch.setattr(onnx_engine, "get_ort_session", fake_get_session)
        onnx_engine.run_onnx_inference("model.onnx", np.zeros((3, 32, 32), dtype=np.float32),
                                       window_size=32, overlap=0, num_classes=2,
                                       activation="softmax",
                                       provider="CUDAExecutionProvider")

        assert captured["provider"] == "CUDAExecutionProvider"
        assert captured["path"] == "model.onnx"


# ============================================================
# inspect_onnx_model — 降级路径
# ============================================================

class TestInspectOnnxModel:
    def test_missing_onnx_package_returns_error(self, monkeypatch):
        """onnx 包非必需依赖, 缺失时应返回 error 而非抛异常"""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "onnx":
                raise ImportError("simulated: no onnx")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        from utils.onnx_engine import inspect_onnx_model
        info = inspect_onnx_model("nonexistent.onnx")
        assert "error" in info
        assert "onnx" in info["error"]

    def test_missing_file_reported(self):
        """本环境无 onnx 包, 走到 error 分支; 文件不存在不应崩"""
        from utils.onnx_engine import inspect_onnx_model
        info = inspect_onnx_model("definitely_not_here.onnx")
        assert "error" in info


# ============================================================
# segment_water_onnx — 前置校验
# ============================================================

class TestSegmentWaterOnnxGuards:
    def test_missing_model_file(self, tmp_path):
        from utils.onnx_engine import segment_water_onnx
        result = segment_water_onnx(
            input_path=str(tmp_path / "img.tif"),
            onnx_model_path=str(tmp_path / "missing.onnx"),
        )
        assert result["success"] is False
        assert "不存在" in result["error"]

    def test_missing_input_file(self, tmp_path):
        """模型存在但输入影像不存在 → 失败而非抛异常"""
        from utils.onnx_engine import segment_water_onnx
        model = tmp_path / "dummy.onnx"
        model.write_bytes(b"\x00" * 2048)

        result = segment_water_onnx(
            input_path=str(tmp_path / "missing.tif"),
            onnx_model_path=str(model),
        )
        assert result["success"] is False
        assert result["error"]

    def test_result_schema(self, tmp_path):
        from utils.onnx_engine import segment_water_onnx
        result = segment_water_onnx(
            input_path=str(tmp_path / "img.tif"),
            onnx_model_path=str(tmp_path / "missing.onnx"),
        )
        for key in ("raster_path", "vector_path", "mask_array", "stats", "success", "error"):
            assert key in result


# ============================================================
# export_to_onnx — 失败路径
# ============================================================

class TestExportToOnnx:
    def test_non_module_input_returns_error(self, tmp_path):
        """传入非 nn.Module 时应返回 success=False 且带 error 说明"""
        from utils.onnx_engine import export_to_onnx
        result = export_to_onnx(
            model="not a torch module",
            output_path=str(tmp_path / "out.onnx"),
            input_shape=(1, 3, 32, 32),
            simplify=False,
            verify=False,
        )
        assert result["success"] is False
        assert result["error"]

    def test_result_schema_on_failure(self, tmp_path):
        from utils.onnx_engine import export_to_onnx
        result = export_to_onnx(
            model=None,
            output_path=str(tmp_path / "out.onnx"),
            input_shape=(1, 3, 32, 32),
            simplify=False,
            verify=False,
        )
        for key in ("success", "output_path", "model_size_mb", "error"):
            assert key in result


# ============================================================
# benchmark_onnx_vs_pytorch — 失败路径
# ============================================================

class TestBenchmark:
    def test_invalid_model_path_records_error(self, tmp_path):
        """模型路径无效时应把错误记进 onnx_error, 而不是抛异常"""
        from utils.onnx_engine import benchmark_onnx_vs_pytorch
        result = benchmark_onnx_vs_pytorch(
            onnx_model_path=str(tmp_path / "nope.onnx"),
            pytorch_model=None,
            input_shape=(1, 3, 16, 16),
            num_runs=1,
            warmup_runs=0,
        )
        assert result["onnx_time_avg_ms"] is None
        assert result["onnx_error"]
        assert result["speedup_ratio"] is None


# ============================================================
# 常量配置
# ============================================================

class TestConstants:
    def test_band_counts_match_sentinel2(self):
        """S2_BAND_COUNTS 用于维度推断, 需覆盖平台实际用到的波段数"""
        from utils.onnx_engine import S2_BAND_COUNTS
        for n in (4, 6, 10, 13):
            assert n in S2_BAND_COUNTS

    def test_window_defaults_are_sane(self):
        from utils.onnx_engine import DEFAULT_WINDOW_SIZE, DEFAULT_OVERLAP
        assert DEFAULT_OVERLAP < DEFAULT_WINDOW_SIZE
        assert DEFAULT_WINDOW_SIZE % 2 == 0
