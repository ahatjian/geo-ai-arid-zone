"""
监督分类训练模块
==================
基于 scikit-learn 的传统机器学习监督分类 (摆脱对 ESA/ESRI 现成产品的依赖)

流程:
  影像多波段 → 特征构建 → 训练样本采集 → 分类器训练 → 整景预测 → 精度评估

分类器 (sklearn):
  - 随机森林 RandomForest (默认, 最稳健)
  - 支持向量机 SVM (RBF 核)
  - K 近邻 KNN
  - 多层感知机 MLP (神经网络)

样本来源:
  - 从参考分类产品 (ESA/ESRI) 自动采样
  - 上传 GeoJSON 样本 (点/面, 带类别属性)
  - 上传 CSV 样本 (x, y, label)

特征:
  - 6 个多光谱波段 (B/G/R/NIR/SWIR1/SWIR2)
  - 可选遥感指数 (NDVI/NDWI/NDBI)

依赖: numpy, scikit-learn, shapely (GeoJSON 解析)
"""

import numpy as np
import warnings
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

# ---- 条件缓存 ----
try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False


def _cache(ttl: int):
    if _HAS_STREAMLIT:
        return st.cache_data(ttl=ttl, show_spinner=True)
    else:
        def noop(func):
            return func
        return noop


# ============================================================
# 分类器配置
# ============================================================

CLASSIFIERS = {
    "random_forest": {
        "name": "随机森林 (Random Forest)",
        "desc": "集成决策树，稳健、抗过拟合，遥感分类首选",
        "params": {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": 2,
            "random_state": 42,
            "n_jobs": -1,
        },
    },
    "svm": {
        "name": "支持向量机 (SVM)",
        "desc": "RBF 核，适合小样本高维特征",
        "params": {
            "kernel": "rbf",
            "C": 10.0,
            "gamma": "scale",
            "probability": True,
        },
    },
    "knn": {
        "name": "K 近邻 (KNN)",
        "desc": "基于距离的非参数分类，简单直观",
        "params": {
            "n_neighbors": 5,
            "weights": "distance",
        },
    },
    "mlp": {
        "name": "多层感知机 (MLP)",
        "desc": "神经网络，可捕捉非线性关系",
        "params": {
            "hidden_layer_sizes": (100, 50),
            "max_iter": 300,
            "random_state": 42,
        },
    },
}

# 特征波段名
FEATURE_BANDS = ["B", "G", "R", "NIR", "SWIR1", "SWIR2"]

# 可选遥感指数特征
INDEX_FEATURES = ["NDVI", "NDWI", "NDBI"]


@dataclass
class SupervisedResult:
    """监督分类结果"""
    model: Optional[object] = None            # 训练好的分类器
    feature_stack: Optional[np.ndarray] = None  # (H, W, n_features)
    prediction: Optional[np.ndarray] = None   # (H, W) 预测分类图
    prediction_proba: Optional[np.ndarray] = None  # (H, W, n_classes) 概率
    train_indices: Optional[np.ndarray] = None  # 训练样本坐标 (N, 2)
    train_labels: Optional[np.ndarray] = None   # 训练样本标签
    accuracy: Optional[Dict] = None           # 精度评估结果
    class_names: List[str] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)


# ============================================================
# 特征构建
# ============================================================

def build_feature_stack(
    bands_dict: Dict[str, np.ndarray],
    add_indices: bool = True,
) -> np.ndarray:
    """
    构建特征堆栈 (H, W, n_features)

    参数:
        bands_dict: {波段名: (H,W) 数组}，含 B/G/R/NIR/SWIR1/SWIR2
        add_indices: 是否添加遥感指数特征 (NDVI/NDWI/NDBI)

    返回:
        feature_stack: (H, W, n_features)
    """
    features = []
    for b in FEATURE_BANDS:
        arr = bands_dict.get(b)
        if arr is None:
            raise ValueError(f"缺少波段: {b}")
        arr = np.asarray(arr, dtype=np.float64)
        # 自动缩放 DN → 反射率
        if np.nanmedian(arr) > 10:
            arr = arr / 10000.0
        features.append(arr)

    if add_indices:
        blue = features[0]
        green = features[1]
        red = features[2]
        nir = features[3]
        swir1 = features[4]

        denom_ndvi = nir + red
        ndvi = np.where(denom_ndvi > 1e-6, (nir - red) / denom_ndvi, 0.0)

        denom_ndwi = green + nir
        ndwi = np.where(denom_ndwi > 1e-6, (green - nir) / denom_ndwi, 0.0)

        denom_ndbi = swir1 + nir
        ndbi = np.where(denom_ndbi > 1e-6, (swir1 - nir) / denom_ndbi, 0.0)

        features.extend([ndvi, ndwi, ndbi])

    stack = np.stack(features, axis=-1)
    return stack.astype(np.float32)


# ============================================================
# 训练样本采集
# ============================================================

def sample_from_reference(
    reference_class: np.ndarray,
    n_samples_per_class: int = 500,
    seed: int = 42,
    skip_zero: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从参考分类图 (如 ESA/ESRI) 中为每类随机采样训练样本

    参数:
        reference_class: (H, W) 参考分类图
        n_samples_per_class: 每类采样数
        seed: 随机种子
        skip_zero: 是否跳过 0 类 (nodata/其他)

    返回:
        (indices, labels)
        - indices: (N, 2) 样本坐标 [row, col]
        - labels: (N,) 样本标签
    """
    reference = np.asarray(reference_class, dtype=np.int16)
    rng = np.random.RandomState(seed)

    indices_list = []
    labels_list = []

    classes = np.unique(reference)
    for cls in classes:
        if skip_zero and cls == 0:
            continue
        ys, xs = np.where(reference == cls)
        if len(ys) == 0:
            continue
        n = min(n_samples_per_class, len(ys))
        sel = rng.choice(len(ys), n, replace=False)
        indices_list.append(np.column_stack([ys[sel], xs[sel]]))
        labels_list.append(np.full(n, cls, dtype=np.int16))

    if not indices_list:
        raise ValueError("参考分类图中没有可采样的类别")

    indices = np.vstack(indices_list)
    labels = np.concatenate(labels_list)
    return indices, labels


def sample_from_geojson(
    geojson_content: bytes,
    class_field: str = "class",
    value_map: Optional[Dict] = None,
) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    从 GeoJSON 样本文件解析训练样本 (点/面几何, 带类别属性)

    参数:
        geojson_content: GeoJSON 字节内容
        class_field: 类别属性字段名
        value_map: 属性值 → 类别号 的映射 (None=直接转 int)

    返回:
        (coords, labels) — coords 为 [(lon, lat), ...], labels 为类别号
    """
    from shapely.geometry import shape
    import json

    data = json.loads(geojson_content.decode("utf-8"))
    features = []
    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
    elif data.get("type") == "Feature":
        features = [data]

    coords = []
    labels = []
    for feat in features:
        props = feat.get("properties", {})
        geom = shape(feat["geometry"])
        # 取类别值
        raw = props.get(class_field)
        if raw is None:
            continue
        if value_map is not None:
            cls = value_map.get(str(raw), value_map.get(raw))
            if cls is None:
                continue
        else:
            cls = int(raw)
        # 点或面 (取代表点: 点=自身, 面=质心)
        if geom.geom_type == "Point":
            coords.append((geom.x, geom.y))
            labels.append(cls)
        elif geom.geom_type in ("Polygon", "MultiPolygon"):
            centroid = geom.centroid
            coords.append((centroid.x, centroid.y))
            labels.append(cls)

    return coords, labels


def sample_from_csv(
    csv_content: bytes,
    x_field: str = "lon",
    y_field: str = "lat",
    label_field: str = "label",
) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    从 CSV 样本文件解析训练样本

    参数:
        csv_content: CSV 字节内容
        x_field: 经度列名
        y_field: 纬度列名
        label_field: 类别列名

    返回:
        (coords, labels)
    """
    import pandas as pd
    from io import BytesIO

    df = pd.read_csv(BytesIO(csv_content))
    coords = list(zip(df[x_field].tolist(), df[y_field].tolist()))
    labels = df[label_field].astype(int).tolist()
    return coords, labels


def coords_to_indices(
    coords: List[Tuple[float, float]],
    transform,
    crs: str = "EPSG:4326",
) -> List[Tuple[int, int]]:
    """
    将经纬度坐标转换为影像行列索引

    参数:
        coords: [(lon, lat), ...]
        transform: rasterio Affine transform
        crs: 影像 CRS

    返回:
        indices: [(row, col), ...] (仅保留在影像范围内的)
    """
    from rasterio.transform import rowcol
    from rasterio.warp import transform as warp_transform

    indices = []
    for lon, lat in coords:
        # 若影像非 4326, 转换坐标
        if crs != "EPSG:4326":
            try:
                lon, lat = warp_transform("EPSG:4326", crs, [lon], [lat])
                lon, lat = lon[0], lat[0]
            except Exception:
                continue
        try:
            row, col = rowcol(transform, lon, lat)
        except Exception:
            continue
        indices.append((int(row), int(col)))

    return indices


# ============================================================
# 分类器训练
# ============================================================

def train_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    classifier: str = "random_forest",
    **params,
):
    """
    训练分类器

    参数:
        features: (N, n_features) 训练样本特征
        labels: (N,) 训练样本标签
        classifier: "random_forest" | "svm" | "knn" | "mlp"
        **params: 分类器参数 (覆盖默认)

    返回:
        model: 训练好的 sklearn 分类器
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier

    cfg = dict(CLASSIFIERS[classifier]["params"])
    cfg.update(params)

    if classifier == "random_forest":
        model = RandomForestClassifier(**cfg)
    elif classifier == "svm":
        model = SVC(**cfg)
    elif classifier == "knn":
        model = KNeighborsClassifier(**cfg)
    elif classifier == "mlp":
        model = MLPClassifier(**cfg)
    else:
        raise ValueError(f"未知分类器: {classifier}")

    model.fit(features, labels)
    return model


def predict_image(model, feature_stack: np.ndarray, batch_size: int = 50000) -> np.ndarray:
    """
    用训练好的模型预测整景影像

    参数:
        model: 训练好的分类器
        feature_stack: (H, W, n_features)
        batch_size: 分批预测大小 (内存保护)

    返回:
        prediction: (H, W) 分类图
    """
    H, W, nf = feature_stack.shape
    flat = feature_stack.reshape(-1, nf)

    # 有效像元 (无 NaN)
    valid = np.isfinite(flat).all(axis=1)
    pred = np.zeros(flat.shape[0], dtype=np.int16)

    valid_flat = flat[valid]
    valid_idx = np.where(valid)[0]

    # 分批预测
    n = len(valid_flat)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_pred = model.predict(valid_flat[start:end])
        pred[valid_idx[start:end]] = batch_pred

    return pred.reshape(H, W)


def predict_image_proba(model, feature_stack: np.ndarray, batch_size: int = 20000) -> np.ndarray:
    """预测整景影像的类别概率 (需要分类器支持 predict_proba)"""
    H, W, nf = feature_stack.shape
    flat = feature_stack.reshape(-1, nf)
    valid = np.isfinite(flat).all(axis=1)

    n_classes = len(model.classes_)
    proba = np.zeros((flat.shape[0], n_classes), dtype=np.float32)

    valid_flat = flat[valid]
    valid_idx = np.where(valid)[0]

    n = len(valid_flat)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        proba[valid_idx[start:end]] = model.predict_proba(valid_flat[start:end])

    return proba.reshape(H, W, n_classes)


# ============================================================
# 精度评估
# ============================================================

def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict:
    """
    分类精度评估 (混淆矩阵 + OA + Kappa + F1)

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表

    返回:
        dict: {confusion_matrix, overall_accuracy, kappa, f1_macro, per_class_f1}
    """
    from sklearn.metrics import (
        confusion_matrix, accuracy_score, cohen_kappa_score, f1_score,
    )

    y_true = np.asarray(y_true, dtype=np.int16)
    y_pred = np.asarray(y_pred, dtype=np.int16)

    n_classes = len(class_names)
    labels = list(range(n_classes))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels)

    # 各类 F1
    per_class_f1 = {}
    f1_per = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for i, name in enumerate(class_names):
        per_class_f1[name] = round(float(f1_per[i]), 4)

    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    return {
        "confusion_matrix": cm,
        "overall_accuracy": round(float(oa), 4),
        "kappa": round(float(kappa), 4),
        "f1_macro": round(float(f1_macro), 4),
        "per_class_f1": per_class_f1,
    }


# ============================================================
# 一站式监督分类
# ============================================================

def assess_supervised_classification(
    bands_dict: Dict[str, np.ndarray],
    train_indices: np.ndarray,
    train_labels: np.ndarray,
    classifier: str = "random_forest",
    add_indices: bool = True,
    test_ratio: float = 0.3,
    seed: int = 42,
    **params,
) -> SupervisedResult:
    """
    一站式监督分类 (训练 → 预测 → 评估)

    参数:
        bands_dict: {波段名: (H,W) 数组}
        train_indices: (N, 2) 训练样本坐标 [row, col]
        train_labels: (N,) 训练样本标签
        classifier: 分类器类型
        add_indices: 是否添加指数特征
        test_ratio: 测试集比例 (用于精度评估)
        seed: 随机种子
        **params: 分类器参数

    返回:
        SupervisedResult
    """
    from sklearn.model_selection import train_test_split

    # 1. 构建特征堆栈
    feature_stack = build_feature_stack(bands_dict, add_indices=add_indices)
    H, W, nf = feature_stack.shape

    # 2. 提取训练样本特征
    rows = train_indices[:, 0]
    cols = train_indices[:, 1]
    # 过滤越界样本
    valid_mask = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    rows = rows[valid_mask]
    cols = cols[valid_mask]
    labels = train_labels[valid_mask]

    sample_features = feature_stack[rows, cols, :]

    # 去除含 NaN 的样本
    finite_mask = np.isfinite(sample_features).all(axis=1)
    sample_features = sample_features[finite_mask]
    labels = labels[finite_mask]

    if len(labels) < 10:
        raise ValueError(f"有效训练样本过少: {len(labels)}")

    # 3. 划分训练/测试集
    if test_ratio > 0 and len(labels) >= 20:
        X_train, X_test, y_train, y_test = train_test_split(
            sample_features, labels, test_size=test_ratio,
            random_state=seed, stratify=labels,
        )
    else:
        X_train, y_train = sample_features, labels
        X_test, y_test = sample_features, labels

    # 4. 训练
    model = train_classifier(X_train, y_train, classifier=classifier, **params)

    # 5. 预测整景影像
    prediction = predict_image(model, feature_stack)

    # 6. 精度评估
    y_pred_test = model.predict(X_test)
    n_classes = len(np.unique(labels))
    class_names = [f"类别{i}" for i in range(max(n_classes, int(np.max(labels)) + 1))]
    accuracy = evaluate_classification(y_test, y_pred_test, class_names)

    # 汇总
    summary = {
        "classifier": CLASSIFIERS[classifier]["name"],
        "n_train_samples": int(len(y_train)),
        "n_test_samples": int(len(y_test)),
        "n_features": nf,
        "overall_accuracy": accuracy["overall_accuracy"],
        "kappa": accuracy["kappa"],
        "f1_macro": accuracy["f1_macro"],
    }

    return SupervisedResult(
        model=model,
        feature_stack=feature_stack,
        prediction=prediction,
        train_indices=np.column_stack([rows, cols]),
        train_labels=labels,
        accuracy=accuracy,
        class_names=class_names,
        summary=summary,
    )
