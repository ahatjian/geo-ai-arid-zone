"""
Geo AI 平台核心模块自动化功能测试
测试: 指数计算 | 趋势分析 | 导出 | 土地覆盖 | 可视化 | 变化检测
API 已与源代码签名对齐
"""
import sys, os
import numpy as np
from io import BytesIO

FAILED = []
PASSED = []

def test(name, fn):
    try:
        fn()
        PASSED.append(name)
        print(f"  ✅ {name}")
    except Exception as e:
        import traceback
        FAILED.append(name)
        print(f"  ❌ {name}: {e}")
        traceback.print_exc()
        # Don't re-raise, continue testing

def section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")

# ============================================================
# 1. utils.indices 测试
# ============================================================
def test_indices():
    from utils.indices import calc_ndvi, calc_mndwi, calc_aweish, calc_evi
    
    rng = np.random.RandomState(42)
    red   = rng.uniform(0.05, 0.15, (256, 256))
    nir   = rng.uniform(0.15, 0.45, (256, 256))
    green = rng.uniform(0.04, 0.12, (256, 256))
    blue  = rng.uniform(0.03, 0.10, (256, 256))
    swir1 = rng.uniform(0.10, 0.30, (256, 256))
    swir2 = rng.uniform(0.08, 0.22, (256, 256))
    
    ndvi = calc_ndvi(nir, red)
    assert ndvi.shape == (256, 256)
    assert -1 <= ndvi.min() <= ndvi.max() <= 1
    
    mndwi = calc_mndwi(green, swir1)
    assert mndwi.shape == (256, 256)
    assert -1 <= mndwi.min() <= mndwi.max() <= 1
    
    aweish = calc_aweish(green, nir, swir1, swir2)
    assert aweish.shape == (256, 256)
    
    evi = calc_evi(nir, red, blue)
    assert evi.shape == (256, 256)
    assert -1 <= evi.min() <= evi.max() <= 1, f"EVI range: {evi.min():.3f} to {evi.max():.3f}"
    
    print(f"    NDVI=[{ndvi.min():.3f},{ndvi.max():.3f}] MNDWI=[{mndwi.min():.3f},{mndwi.max():.3f}] EVI=[{evi.min():.3f},{evi.max():.3f}]")

# ============================================================
# 2. utils.trend 测试 (Sen+MK)
# ============================================================
def test_trend():
    from utils.trend import calc_sen_mk_trend
    
    # 上升趋势
    ts_up = np.array([0.1, 0.15, 0.18, 0.22, 0.25, 0.30, 0.35, 0.40])
    result = calc_sen_mk_trend(ts_up)
    # 返回 dict: {slope, trend, p_value, significant, intercept, n_valid}
    assert result['slope'] > 0, f"Expected positive slope, got {result['slope']}"
    assert result['trend'] in ["increasing", "decreasing", "no trend"]
    print(f"    Up: slope={result['slope']:.4f} p={result['p_value']:.4f} trend={result['trend']} sig={result['significant']}")
    
    # 无趋势
    ts_flat = np.array([0.2, 0.21, 0.19, 0.22, 0.20, 0.21, 0.19, 0.20])
    r2 = calc_sen_mk_trend(ts_flat)
    print(f"    Flat: slope={r2['slope']:.4f} p={r2['p_value']:.4f} trend={r2['trend']} sig={r2['significant']}")
    
    # 多像元
    ts_batch = np.array([
        [0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        [0.3, 0.28, 0.25, 0.22, 0.2, 0.15],
        [0.2, 0.21, 0.19, 0.22, 0.20, 0.21],
    ])
    for i in range(3):
        r = calc_sen_mk_trend(ts_batch[i])
        print(f"    Pixel{i}: slope={r['slope']:.4f} p={r['p_value']:.4f} trend={r['trend']}")

# ============================================================
# 3. utils.export 测试
# ============================================================
def test_export():
    from utils.export import export_csv, export_stats_csv, export_numpy, make_download_label
    
    # make_download_label(prefix, ext, timestamp=True)
    label = make_download_label("test_prefix", "csv")
    assert "_" in label or "test_prefix" in label
    print(f"    Download label: {label}")
    
    # export_csv expects DataFrame/Dict/List[Dict] not ndarray
    data = [{"col_a": float(i), "col_b": float(i*2), "col_c": float(i*3)} for i in range(100)]
    result_path, result_bytes = export_csv(data)
    assert len(result_bytes) > 0
    csv_str = result_bytes.decode() if isinstance(result_bytes, bytes) else str(result_bytes)
    assert "col_a" in csv_str
    print(f"    CSV: {len(csv_str)} bytes, header OK")
    
    # export_stats_csv(stats: Dict, output_path=None, label='...') -> Tuple[str, bytes]
    stats_data = {"水体面积(km2)": 12.5, "植被覆盖率(%)": 45.2, "变化面积(km2)": 3.1}
    _, stats_bytes = export_stats_csv(stats_data, label="测试统计")
    assert "水体面积" in stats_bytes.decode()
    print(f"    Stats CSV: OK")
    
    # NumPy: export_numpy(arr, output_path=None, compress=True) -> Tuple[str, bytes]
    arr = np.array([1.0, 2.0, 3.0])
    _, np_bytes = export_numpy(arr)
    assert len(np_bytes) > 0
    print(f"    NumPy: {len(np_bytes)} bytes")

# ============================================================
# 4. utils.landcover 测试
# ============================================================
def test_landcover():
    from utils.landcover import (
        ESA_CLASSES, ESRI_CLASSES, ARID6_CLASSES,
        ESA_TO_ARID6, ESRI_TO_ARID6,
        _esa_tile_name, compute_landcover_stats
    )
    
    # 类别数量
    assert len(ESA_CLASSES) >= 11
    assert len(ARID6_CLASSES) == 6
    print(f"    ESA={len(ESA_CLASSES)} classes → Arid6={len(ARID6_CLASSES)} classes")
    
    # ESA tile name (返回 "N39E090" 格式)
    tile = _esa_tile_name(40.0, 90.0)
    assert len(tile) > 0
    print(f"    ESA tile: {tile}")
    
    # compute_landcover_stats(class_array, class_names, class_colors, pixel_size_m=10.0)
    fake_class_map = np.random.choice([0, 10, 20, 30], (500, 500))
    ESA_NAMES = {k: v['name'] for k, v in ESA_CLASSES.items()}
    ESA_COLORS = {k: v['color'] for k, v in ESA_CLASSES.items()}
    stats = compute_landcover_stats(fake_class_map, ESA_NAMES, ESA_COLORS, pixel_size_m=100.0)
    assert len(stats) > 0
    print(f"    Stats: {len(stats)} classes, e.g. {stats[0]['class_name']}: {stats[0]['area_km2']:.2f} km2")

# ============================================================
# 5. utils.visualization 测试
# ============================================================
def test_visualization():
    from utils.visualization import (
        render_index, render_change_map, render_classification,
        plot_time_series, plot_dual_time_series, plot_histogram
    )
    from PIL import Image
    
    # render_index → PIL Image
    ndvi = np.random.uniform(-0.2, 0.8, (200, 200))
    img = render_index(ndvi, vmin=-0.2, vmax=0.8, cmap="RdYlGn")
    assert isinstance(img, Image.Image), f"Got {type(img)}"
    print(f"    render_index: PIL Image {img.size}")
    
    # render_change_map → PIL Image
    delta = np.random.uniform(-0.5, 0.5, (200, 200))
    img2 = render_change_map(delta)
    assert isinstance(img2, Image.Image)
    print(f"    render_change_map: PIL Image {img2.size}")
    
    # render_classification → PIL Image
    classes = np.random.choice([1, 2, 3, 4, 5, 6], (200, 200))
    class_names = ["水体", "植被", "裸地", "建设用地", "农田", "矿区"]
    class_colors = ["#1f77b4", "#2ca02c", "#d4a76a", "#d62728", "#ff7f0e", "#8c564b"]
    img3 = render_classification(classes, class_names, class_colors, "测试")
    assert isinstance(img3, Image.Image)
    print(f"    render_classification: PIL Image {img3.size}")
    
    # plot_time_series
    dates = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"]
    values = [0.2, 0.3, 0.4, 0.35, 0.5]
    fig = plot_time_series(dates, values, title="测试时序")
    import plotly.graph_objects as go
    assert isinstance(fig, go.Figure)
    print(f"    plot_time_series: Plotly Figure OK")

# ============================================================
# 6. 变化检测函数测试
# ============================================================
def test_change_detection():
    def classify_multilevel(delta, thresholds=[-0.3, -0.15, -0.05, 0.05, 0.15, 0.3]):
        result = np.zeros_like(delta, dtype=int)
        result[delta < thresholds[0]] = -3
        for i in range(len(thresholds)-1):
            mask = (delta >= thresholds[i]) & (delta < thresholds[i+1])
            result[mask] = -3 + i + 1
        result[delta >= thresholds[-1]] = 3
        return result
    
    from utils.visualization import render_multilevel_change, render_change_overlay, plot_multilevel_change_stacked_bar
    from PIL import Image
    
    delta = np.random.uniform(-0.5, 0.5, (200, 200))
    levels = classify_multilevel(delta)
    
    unique_vals = np.unique(levels)
    assert -3 <= unique_vals.min() and unique_vals.max() <= 3
    print(f"    7-level: unique={unique_vals}")
    
    img = render_multilevel_change(levels)
    assert isinstance(img, Image.Image)
    print(f"    render_multilevel_change: PIL OK")
    
    # RGB overlay
    rgb = np.random.uniform(0, 1, (200, 200, 3))
    img2 = render_change_overlay(rgb, levels, alpha=0.5)
    assert isinstance(img2, Image.Image)
    print(f"    render_change_overlay: PIL OK")
    
    # Stacked bar: plot_multilevel_change_stacked_bar(level_counts, total_pixels, title, height)
    counts = {k: int(np.sum(levels == k)) for k in range(-3, 4)}
    total_pixels = 200 * 200
    fig = plot_multilevel_change_stacked_bar(counts, total_pixels)
    import plotly.graph_objects as go
    assert isinstance(fig, go.Figure)
    print(f"    plot_multilevel_change_stacked_bar: Plotly OK")

# ============================================================
# 7. 配置测试
# ============================================================
def test_config():
    import config
    
    # STUDY_AREAS is dict with Chinese names as keys
    assert len(config.STUDY_AREAS) >= 6
    names = list(config.STUDY_AREAS.keys())
    print(f"    Study areas: {len(names)} — {', '.join(names[:3])}...")
    
    for name, area in config.STUDY_AREAS.items():
        assert 'bbox' in area
        assert 'center' in area
        break
    
    # COLLECTIONS is dict
    assert len(config.COLLECTIONS) >= 2
    col_names = list(config.COLLECTIONS.keys())
    print(f"    Collections: {', '.join(col_names)}")

# ============================================================
# 8. AI引擎导入测试
# ============================================================
def test_ai_engine():
    from utils.ai_engine import segment_water_ai, build_band_order, get_model_info
    
    info = get_model_info()
    assert isinstance(info, dict)
    print(f"    get_model_info: {len(info)} keys: {list(info.keys())[:4]}")
    
    # build_band_order(band_red, band_green, band_blue, band_nir) → List[int]
    order = build_band_order(band_red=1, band_green=2, band_blue=3, band_nir=4)
    assert len(order) == 4
    print(f"    build_band_order: {order}")

# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Geo AI 平台核心模块自动化测试 v2")
    print("=" * 60)
    
    section("1. 指数计算 (indices.py)")
    test("NDVI/EVI/MNDWI/AWEIsh", test_indices)
    
    section("2. Sen+MK 趋势分析 (trend.py)")
    test("趋势检验", test_trend)
    
    section("3. 统一导出 (export.py)")
    test("CSV/NP/Stats/Label", test_export)
    
    section("4. 土地覆盖 (landcover.py)")
    test("类别映射+统计", test_landcover)
    
    section("5. 可视化 (visualization.py)")
    test("渲染+时序图", test_visualization)
    
    section("6. 变化检测 (7级分类)")
    test("7级分类+可视化", test_change_detection)
    
    section("7. 配置 (config.py)")
    test("研究区+数据源", test_config)
    
    section("8. AI引擎 (ai_engine.py)")
    test("模型信息+波段映射", test_ai_engine)
    
    print(f"\n{'='*60}")
    print(f"  结果: {len(PASSED)}/{len(PASSED)+len(FAILED)} 通过")
    if FAILED:
        print(f"  ❌ 失败: {', '.join(FAILED)}")
        sys.exit(1)
    else:
        print(f"  🎉 全部通过! (8/8)")
    print(f"{'='*60}")
