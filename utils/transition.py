"""
土地覆盖转移矩阵模块
======================
双时相土地覆盖分类对比，生成土地利用转移矩阵

功能:
  - 转移矩阵计算 (面积 km² 或像元数)
  - 各类型净变化 (转入/转出/净变化)
  - 主要转移方向识别
  - 转移矩阵热力图

应用: 土地利用变化分析、生态环境变化评估、科研论文标准产出

依赖: numpy, matplotlib
"""

import numpy as np
import warnings
from typing import Optional, Dict, List, Tuple

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
# 转移矩阵计算
# ============================================================

def compute_transition_matrix(
    class_t1: np.ndarray,
    class_t2: np.ndarray,
    n_classes: int = 6,
    pixel_size_m: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算土地覆盖转移矩阵

    参数:
        class_t1: (H, W) T1 时相分类图 (值 0 ~ n_classes-1)
        class_t2: (H, W) T2 时相分类图 (值 0 ~ n_classes-1)
        n_classes: 类别数
        pixel_size_m: 像元大小 (m)

    返回:
        (area_matrix, pixel_matrix)
        - area_matrix: (n_classes, n_classes) 转移面积 (km²), 行=T1, 列=T2
        - pixel_matrix: (n_classes, n_classes) 转移像元数
    """
    class_t1 = np.asarray(class_t1, dtype=np.int16)
    class_t2 = np.asarray(class_t2, dtype=np.int16)

    if class_t1.shape != class_t2.shape:
        raise ValueError(
            f"两个时相分类图尺寸不一致: {class_t1.shape} vs {class_t2.shape}"
        )

    pixel_area_km2 = (pixel_size_m ** 2) / 1e6

    pixel_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for i in range(n_classes):
        for j in range(n_classes):
            pixel_matrix[i, j] = np.sum((class_t1 == i) & (class_t2 == j))

    area_matrix = pixel_matrix * pixel_area_km2

    return area_matrix, pixel_matrix


def net_change(
    area_matrix: np.ndarray,
    class_names: List[str],
) -> List[Dict]:
    """
    计算各土地类型净变化 (转入 - 转出)

    参数:
        area_matrix: (n, n) 转移面积矩阵 (行=T1, 列=T2)
        class_names: 类别名称列表

    返回:
        list: 每个类别的 {name, gain, loss, net, change_pct}
    """
    n = area_matrix.shape[0]
    results = []
    for i in range(n):
        loss = float(np.sum(area_matrix[i, :]) - area_matrix[i, i])  # 转出
        gain = float(np.sum(area_matrix[:, i]) - area_matrix[i, i])  # 转入
        net = gain - loss
        initial = float(np.sum(area_matrix[i, :]))
        change_pct = (net / initial * 100) if initial > 0 else 0.0

        results.append({
            "name": class_names[i],
            "initial_area_km2": round(initial, 2),
            "gain_km2": round(gain, 2),
            "loss_km2": round(loss, 2),
            "net_km2": round(net, 2),
            "change_pct": round(change_pct, 2),
        })
    return results


def find_major_transitions(
    area_matrix: np.ndarray,
    class_names: List[str],
    top_n: int = 8,
) -> List[Dict]:
    """
    识别主要土地转移方向 (非对角线, 按面积排序)

    参数:
        area_matrix: (n, n) 转移面积矩阵 (行=T1, 列=T2)
        class_names: 类别名称列表
        top_n: 返回前 N 个主要转移

    返回:
        list: [{from, to, area_km2}, ...] 按面积降序
    """
    n = area_matrix.shape[0]
    transitions = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            area = area_matrix[i, j]
            if area > 0:
                transitions.append({
                    "from": class_names[i],
                    "to": class_names[j],
                    "area_km2": round(float(area), 2),
                })

    transitions.sort(key=lambda x: x["area_km2"], reverse=True)
    return transitions[:top_n]


def plot_transition_heatmap(
    area_matrix: np.ndarray,
    class_names: List[str],
    title: str = "土地覆盖转移矩阵",
    figsize: Tuple[int, int] = (9, 7),
    return_fig: bool = False,
):
    """
    绘制转移矩阵热力图

    参数:
        area_matrix: (n, n) 转移面积矩阵 (km²)
        class_names: 类别名称列表
        title: 标题
        figsize: 图大小
        return_fig: 是否返回 figure

    返回:
        matplotlib figure (return_fig=True 时)
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    # 用 log 尺度增强可视化 (转移面积差异大)
    log_matrix = np.log1p(area_matrix)

    im = ax.imshow(log_matrix, cmap="YlOrRd", aspect="auto")

    # 标注数值
    n = area_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            val = area_matrix[i, j]
            if val > 0:
                label = f"{val:.1f}" if val < 100 else f"{val:.0f}"
                color = "white" if val > np.nanmax(area_matrix) * 0.3 else "black"
                ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("T2 时相 (后)", fontsize=11)
    ax.set_ylabel("T1 时相 (前)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log(1 + 面积 km²)")
    plt.tight_layout()

    if return_fig:
        return fig
    else:
        plt.show()
        plt.close()


def plot_net_change_bar(
    net_change_list: List[Dict],
    title: str = "土地类型净变化",
    figsize: Tuple[int, int] = (9, 5),
    return_fig: bool = False,
):
    """
    绘制各土地类型净变化柱状图

    参数:
        net_change_list: net_change() 返回的列表
        title: 标题
        return_fig: 是否返回 figure
    """
    import matplotlib.pyplot as plt

    names = [x["name"] for x in net_change_list]
    nets = [x["net_km2"] for x in net_change_list]

    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in nets]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(names, nets, color=colors)
    ax.axhline(0, color="#333", linewidth=0.8)

    for bar, v in zip(bars, nets):
        va = "bottom" if v >= 0 else "top"
        offset = max(abs(v) * 0.02, 0.1)
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + (offset if v >= 0 else -offset),
                f"{v:+.1f}", ha="center", va=va, fontsize=9)

    ax.set_ylabel("净变化面积 (km²)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if return_fig:
        return fig
    else:
        plt.show()
        plt.close()


@_cache(ttl=600)
def analyze_transition(
    class_t1: np.ndarray,
    class_t2: np.ndarray,
    class_names: List[str],
    pixel_size_m: float = 10.0,
    top_n: int = 8,
) -> Dict:
    """
    一站式土地覆盖转移分析

    参数:
        class_t1: (H, W) T1 分类图
        class_t2: (H, W) T2 分类图
        class_names: 类别名称列表
        pixel_size_m: 像元大小
        top_n: 主要转移方向数量

    返回:
        dict: {
            "area_matrix", "pixel_matrix", "net_change",
            "major_transitions", "total_change_km2"
        }
    """
    n_classes = len(class_names)
    area_matrix, pixel_matrix = compute_transition_matrix(
        class_t1, class_t2, n_classes=n_classes, pixel_size_m=pixel_size_m
    )

    nc = net_change(area_matrix, class_names)
    mt = find_major_transitions(area_matrix, class_names, top_n=top_n)

    # 总变化面积 (非对角线之和)
    total_change = float(np.sum(area_matrix) - np.trace(area_matrix))

    return {
        "area_matrix": area_matrix,
        "pixel_matrix": pixel_matrix,
        "net_change": nc,
        "major_transitions": mt,
        "total_change_km2": round(total_change, 2),
    }
