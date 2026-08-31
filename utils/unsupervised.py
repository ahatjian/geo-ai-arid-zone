"""
非监督分类模块 — KMeans/ISODATA 聚类
======================================
不依赖训练样本, 自动将影像像元聚类为 K 类地物。

方法:
  - KMeans: 经典 K 均值聚类 (sklearn)
  - ISODATA 风格: 迭代自组织聚类 (KMeans 变体 + 合并/分裂启发)

适用于:
  - 无样本时的快速土地利用概览
  - 分类结果与监督分类交叉验证
  - 影像分割前的初步聚类
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def kmeans_classify(
    bands_data: np.ndarray,
    n_classes: int = 6,
    max_iter: int = 100,
    random_state: int = 42,
    sample_ratio: float = 0.1,
) -> Dict[str, np.ndarray]:
    """
    KMeans 非监督分类。

    原理: 将像元光谱向量聚类为 K 类, 最小化类内距离平方和。

    参数:
        bands_data: (B, H, W) 多波段数组
        n_classes: 聚类数 K
        max_iter: 最大迭代次数
        random_state: 随机种子 (可复现)
        sample_ratio: 采样比例 (大影像用子集加速, 1.0=全部)

    返回:
        dict: {
            "classification": (H, W) uint8 分类图 (0..K-1),
            "centers": (K, B) 各类中心光谱,
            "sizes": (K,) 各类像元数,
            "inertia": float 类内距离平方和 (越小越紧凑),
            "method": str
        }
    """
    from sklearn.cluster import KMeans

    bands_data = np.asarray(bands_data, dtype=np.float64)
    if bands_data.ndim != 3:
        raise ValueError(f"需要 3D 数组 (B,H,W), 实际: {bands_data.ndim}D")
    B, H, W = bands_data.shape

    # 展平: (N, B)
    flat = bands_data.reshape(B, -1).T  # (H*W, B)

    # 处理 NaN: 移除无效像元
    valid_mask = np.all(np.isfinite(flat), axis=1)
    valid = flat[valid_mask]

    if len(valid) < n_classes * 10:
        raise ValueError(f"有效像元过少: {len(valid)}")

    # 采样加速
    if sample_ratio < 1.0 and len(valid) > 10000:
        rng = np.random.default_rng(random_state)
        n_sample = max(int(len(valid) * sample_ratio), n_classes * 20)
        idx = rng.choice(len(valid), n_sample, replace=False)
        sample = valid[idx]
    else:
        sample = valid

    # KMeans 聚类
    km = KMeans(
        n_clusters=n_classes,
        max_iter=max_iter,
        random_state=random_state,
        n_init=10,
    )
    km.fit(sample)

    # 预测全部像元
    labels = km.predict(valid)

    # 还原到图像网格 (先扁平化再 reshape)
    classification_flat = np.full(flat.shape[0], -1, dtype=np.int16)
    classification_flat[valid_mask] = labels
    classification = classification_flat.reshape(H, W)

    # 统计
    sizes = np.bincount(labels, minlength=n_classes)

    return {
        "classification": classification.astype(np.int16),
        "centers": km.cluster_centers_,
        "sizes": sizes,
        "inertia": float(km.inertia_),
        "method": f"KMeans (K={n_classes})",
    }


def kmeans_feature_stack(
    bands_data: np.ndarray,
    add_indices: bool = True,
) -> np.ndarray:
    """
    构建聚类特征栈: 原始波段 + 可选光谱指数。

    增加 NDVI/MNDWI/NDBI 可显著提升植被/水体/建筑的可分性。

    参数:
        bands_data: (B, H, W) 标准 6 波段 [B,G,R,NIR,SWIR1,SWIR2]
        add_indices: 是否加入 NDVI/MNDWI/NDBI

    返回:
        (B+K, H, W) 特征栈
    """
    bands_data = np.asarray(bands_data, dtype=np.float64)
    if bands_data.shape[0] < 4:
        return bands_data

    B, G, R, NIR = bands_data[0], bands_data[1], bands_data[2], bands_data[3]
    SWIR1 = bands_data[4] if bands_data.shape[0] > 4 else NIR

    stack = [bands_data]

    if add_indices:
        denom_ndvi = NIR + R
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = np.where(denom_ndvi > 1e-6, (NIR - R) / denom_ndvi, 0.0)
        denom_mndwi = G + SWIR1
        with np.errstate(divide="ignore", invalid="ignore"):
            mndwi = np.where(denom_mndwi > 1e-6, (G - SWIR1) / denom_mndwi, 0.0)
        denom_ndbi = SWIR1 + NIR
        with np.errstate(divide="ignore", invalid="ignore"):
            ndbi = np.where(denom_ndbi > 1e-6, (SWIR1 - NIR) / denom_ndbi, 0.0)
        stack.append(np.stack([ndvi, mndwi, ndbi]))

    return np.concatenate(stack, axis=0)


def auto_describe_classes(
    centers: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> List[str]:
    """
    根据聚类中心光谱特征自动推断类别名 (启发式)。

    用 NDVI/NIR 特征推断: 高 NIR+NDVI=植被, 低 NIR+高 SWIR=裸地等。

    参数:
        centers: (K, B) 聚类中心 (标准 6 波段)
        class_names: 自定义类别名 (None=自动推断)

    返回:
        list: 每类的描述名
    """
    if class_names is not None and len(class_names) == len(centers):
        return list(class_names)

    names = []
    for c in centers:
        if len(c) >= 4:
            nir, red = c[3], c[2]
            ndvi = (nir - red) / (nir + red + 1e-6)
            # 水体: 全波段低反射 (可见光/近红外均低)
            mean_vis = np.mean(c[:3]) if len(c) >= 3 else red
            if ndvi > 0.4:
                names.append("茂密植被")
            elif ndvi > 0.2:
                names.append("稀疏植被/农田")
            elif mean_vis < 0.12 and nir < 0.12:
                names.append("水体")
            else:
                swir = c[4] if len(c) > 4 else nir
                if swir > 0.3:
                    names.append("裸地/荒漠")
                else:
                    names.append("混合地表")
        else:
            names.append(f"类别{len(names)+1}")
    return names
