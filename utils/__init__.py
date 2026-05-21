"""
Geo AI 干旱区遥感智能分析平台 — 工具模块
"""

from .indices import calc_ndvi, calc_mndwi, calc_evi, calc_aweish, load_bands_from_geotiff
from .visualization import (
    render_classification, render_ndvi, render_mndwi,
    render_multilevel_change, render_change_overlay,
    plot_multilevel_change_stacked_bar, plot_trend_scatter,
)
from .landcover import (
    ESA_CLASSES, ARID6_CLASSES, ESRI_CLASSES,
    ESA_TO_ARID6, ESRI_TO_ARID6,
    esa_to_arid6, esri_to_arid6,
    get_landcover_for_study_area, compute_landcover_stats,
    get_tile_list_for_area, export_landcover_geotiff,
)
from .ai_engine import (
    segment_water_ai, segment_water_ai_batch,
    build_band_order, get_model_info,
    SENTINEL2_BAND_ORDER, DEFAULT_SEGMENT_PARAMS,
)
from .trend import (
    calc_sen_mk_trend, calc_sen_mk_trend_pixelwise,
    calc_mean_timeseries_trend, theil_sen_slope,
)
from .export import (
    export_raster_geotiff, export_index_geotiff, export_mask_geotiff,
    export_classification_geotiff,
    export_csv, export_stats_csv, export_timeseries_csv,
    export_figure, export_array_as_image, export_comparison_figure,
    export_timeseries_plot,
    raster_to_geojson_polygons, export_change_polygons_geojson,
    export_numpy, export_batch, make_download_label,
)
from .onnx_engine import (
    check_onnx_available, get_ort_session,
    export_to_onnx, run_onnx_inference,
    segment_water_onnx, benchmark_onnx_vs_pytorch,
    inspect_onnx_model,
    DEFAULT_WINDOW_SIZE, DEFAULT_OVERLAP,
)
from .error_handler import (
    with_retry, safe_execute,
    StreamlitErrorBoundary, streamlit_safe,
    StreamlitProgress, check_dependencies,
    get_memory_usage_mb, safe_cleanup,
)
from .drought import (
    DROUGHT_CATEGORIES, SPI_THRESHOLDS, VCI_THRESHOLDS,
    classify_drought, classify_spi, classify_vci,
    calc_spi, calc_spi_pixelwise,
    calc_pet_thornthwaite, calc_spei,
    calc_vci, calc_vci_pixelwise,
    calc_tci, calc_tci_pixelwise,
    calc_vhi, calc_vhi_pixelwise,
    calc_nddi, calc_ndwi_s2,
    calc_ndvi_anomaly, calc_ndvi_anomaly_pixelwise,
    calc_tvdi,
    calc_composite_drought_index,
    compute_drought_stats, compute_drought_index_stats,
    analyze_drought_remote,
)
