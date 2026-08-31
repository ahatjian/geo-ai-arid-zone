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
    savgol_smooth, smooth_ndvi_stack,
    stl_decompose,
)
from .forecast import (
    ForecastResult,
    prepare_ndvi_timeseries, split_train_test, evaluate_forecast,
    forecast_sarima, forecast_lstm, forecast_holt_winters,
    forecast_drought_trend,
    plot_forecast, plot_forecast_comparison,
    analyze_drought_forecast,
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
from .desertification import (
    DESERTIFICATION_LEVELS, DesertificationResult,
    calc_albedo_s2, calc_albedo_landsat,
    calc_tgsi, calc_ndmi, calc_ddi,
    classify_desertification, compute_desertification_stats,
    assess_desertification, assess_desertification_cached,
    analyze_desertification_trend, get_desertification_colormap,
)
from .salinity import (
    SALINITY_LEVELS, SalinityResult, DEFAULT_NDSI_THRESHOLDS,
    calc_si, calc_si1, calc_si2,
    calc_ndsi_salinity, calc_bi, calc_ndvi_mask,
    classify_salinity, compute_salinity_stats,
    get_salinity_colormap, assess_salinity, assess_salinity_cached,
)
from .lst import (
    THERMAL_LEVELS, LSTResult, DEFAULT_THERMAL_THRESHOLDS,
    read_lst_array, kelvin_to_celsius, classify_thermal,
    compute_lst_stats, get_thermal_colormap,
    compute_lst_ndvi_relation, assess_thermal, assess_thermal_cached,
)
from .spectral import (
    BAND_NAMES, PRESET_INDICES, PRESET_INDEX_NAMES, ALLOWED_NUMPY_FUNCS,
    get_band_arrays, evaluate_band_math,
    get_preset_index, compute_index_stats,
)
from .transition import (
    compute_transition_matrix, net_change,
    find_major_transitions, plot_transition_heatmap,
    plot_net_change_bar, analyze_transition,
)
from .aoi import (
    parse_geojson_bbox, validate_bbox, render_aoi_selector,
)
from .evapotranspiration import (
    ET_LEVELS, ETResult, DEFAULT_ET_THRESHOLDS,
    calc_fvc, calc_emissivity, calc_atmospheric_emissivity,
    calc_net_radiation, calc_soil_heat_flux, calc_sensible_heat,
    calc_latent_heat, calc_et_daily,
    classify_et, compute_et_stats, get_et_colormap,
    assess_et, assess_et_cached,
)
from .supervised import (
    CLASSIFIERS, FEATURE_BANDS, INDEX_FEATURES, SupervisedResult,
    build_feature_stack, sample_from_reference,
    sample_from_geojson, sample_from_csv, coords_to_indices,
    train_classifier, predict_image, predict_image_proba,
    evaluate_classification, assess_supervised_classification,
)
from .vector import (
    VECTOR_FORMATS, DEFAULT_AREA_CRS,
    raster_to_gdf, gdf_to_geojson, gdf_to_shapefile, gdf_to_kml,
    raster_to_vector, raster_to_vector_from_file,
    compute_class_areas, summarize_vector,
)
from .results_store import (
    KIND_EXT, TEXT_KINDS, BINARY_KINDS,
    save_result, save_result_file, list_results,
    get_result, get_result_path, load_result,
    delete_result, clear_results, package_results, get_store_info,
)
from .ai_insight import (
    generate_ai_insight, explain_metrics,
    summarize_insight, get_insight_history,
    is_ai_available,
)
from .save_ui import (
    render_save_button, render_save_csv_button,
)
from .preprocess import (
    apply_s2_cloud_mask, apply_landsat_cloud_mask,
    mask_clouds, resample_array, normalize_to_uint8,
    cloud_cover_fraction, mask_stats,
)
from .pdf_report import (
    generate_report_pdf, report_to_pdf_download,
)
from .composite import (
    composite_ndvi_monthly, composite_series_by_month, merge_ndvi_max,
)
from .image_processing import (
    pca_transform, pca_rgb_composite,
    spatial_filter, contrast_enhance, ihs_fusion, enhance_report,
)
from .unsupervised import (
    kmeans_classify, kmeans_feature_stack, auto_describe_classes,
)
from .atmospheric import (
    dos_correction, estimate_dark_pixel, dos_quality_report,
    radiometric_calibration, toa_reflectance, relative_normalization,
    preprocess_pipeline,
)
from .bfast import (
    detect_breaks, chow_test, detect_vegetation_breaks,
    summarize_breaks,
)
from .cryosphere import (
    SNOW_COVER_CLASSES, GLACIER_CLASSES, CryosphereResult,
    calc_ndsi, calc_ndsi_nir, calc_snow_cover,
    extract_glacier_mask, estimate_snow_line,
    analyze_frozen_ground, compute_snow_cover_stats,
    compute_glacier_stats, assess_cryosphere,
)
from .agri_drought import (
    AGRI_DROUGHT_LEVELS, AgriDroughtResult,
    calc_cwsi_ndvi, calc_smi_swir, calc_smi_combined,
    calc_mpdi, classify_agri_drought,
    estimate_irrigation_demand, compute_agri_stats,
    assess_agri_drought,
)
from .animation import (
    create_timeseries_animation, create_multi_index_animation,
    create_trend_animation,
)
from .ecology import (
    ECO_SECURITY_LEVELS, EcoSecurityResult,
    calc_psi, calc_ssi, calc_rsi, calc_esi,
    classify_eco_security, compute_eco_stats,
    assess_eco_security,
)
