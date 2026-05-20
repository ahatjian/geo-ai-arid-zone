"""
可视化辅助模块
封装 matplotlib/plotly 图表生成函数，供各 Streamlit 页面复用
依赖: matplotlib, plotly, numpy, PIL
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from PIL import Image
import io
import warnings

warnings.filterwarnings("ignore")

# ============================================
# 全局样式配置
# ============================================

# 中文字体设置 (Windows)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# 色带配置
COLORMAPS = {
    "NDVI": "RdYlGn",
    "EVI": "YlGn",
    "MNDWI": "Blues",
    "AWEIsh": "coolwarm",
    "water_mask": "Blues",
    "change_diff": "RdYlGn",
    "land_cover": "tab10",
}

# ============================================
# 指数渲染为 PIL Image
# ============================================

def render_index(
    data: np.ndarray,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "RdYlGn",
    figsize: tuple = (8, 6),
    dpi: int = 100,
    colorbar_label: str = "",
    title: str = "",
) -> Image.Image:
    """
    将指数数组渲染为彩色图像

    参数:
        data: 2D numpy 数组
        vmin, vmax: 色彩映射范围
        cmap: matplotlib 色带名
        figsize: 图表尺寸
        dpi: 分辨率
        colorbar_label: 色标标签
        title: 图表标题

    返回:
        PIL.Image
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=13)
    ax.axis("off")

    if colorbar_label:
        cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
        cbar.set_label(colorbar_label, fontsize=10)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return Image.open(buf)


def render_ndvi(ndvi: np.ndarray, title: str = "NDVI 植被指数") -> Image.Image:
    """快捷渲染 NDVI"""
    return render_index(
        ndvi,
        vmin=-0.2,
        vmax=0.8,
        cmap="RdYlGn",
        colorbar_label="NDVI",
        title=title,
    )


def render_evi(evi: np.ndarray, title: str = "EVI 增强植被指数") -> Image.Image:
    """快捷渲染 EVI"""
    return render_index(
        evi,
        vmin=-0.2,
        vmax=0.8,
        cmap="YlGn",
        colorbar_label="EVI",
        title=title,
    )


def render_mndwi(mndwi: np.ndarray, title: str = "MNDWI 水体指数") -> Image.Image:
    """快捷渲染 MNDWI"""
    return render_index(
        mndwi,
        vmin=-1.0,
        vmax=1.0,
        cmap="Blues",
        colorbar_label="MNDWI",
        title=title,
    )


def render_aweish(aweish: np.ndarray, title: str = "AWEIsh 水体指数") -> Image.Image:
    """快捷渲染 AWEIsh"""
    vmin_val = np.nanpercentile(aweish, 2)
    vmax_val = np.nanpercentile(aweish, 98)
    return render_index(
        aweish,
        vmin=vmin_val,
        vmax=vmax_val,
        cmap="coolwarm",
        colorbar_label="AWEIsh",
        title=title,
    )


# ============================================
# 水体掩膜可视化
# ============================================

def render_water_mask(
    water_mask: np.ndarray,
    rgb_image: Image.Image = None,
    title: str = "水体提取结果",
) -> Image.Image:
    """
    渲染水体掩膜，可选叠加在 RGB 影像上

    参数:
        water_mask: 二值掩膜 (1=水体, 0=非水体)
        rgb_image: RGB 底图 PIL.Image (可选)
        title: 标题

    返回:
        PIL.Image
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if rgb_image is not None:
        rgb_array = np.array(rgb_image)
        ax.imshow(rgb_array)

    # 叠加水体掩膜 (蓝色半透明)
    water_overlay = np.zeros((*water_mask.shape, 4))
    water_overlay[water_mask == 1] = [0, 0.4, 1.0, 0.5]  # RGBA 蓝色半透明
    ax.imshow(water_overlay)

    ax.set_title(title, fontsize=13)
    ax.axis("off")

    # 图例
    legend_elements = [
        Patch(facecolor=(0, 0.4, 1.0, 0.5), label="水体"),
        Patch(facecolor="lightgray", label="非水体"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return Image.open(buf)


# ============================================
# 时序分析图表 (Plotly)
# ============================================

def plot_time_series(
    dates: list,
    values: list,
    y_label: str = "Index Value",
    title: str = "时序变化",
    line_color: str = "#1f77b4",
    height: int = 400,
):
    """
    Plotly 时序折线图

    参数:
        dates: 日期列表
        values: 值列表
        y_label: Y轴标签
        title: 图表标题
        line_color: 线条颜色
        height: 图表高度

    返回:
        plotly Figure
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=values,
            mode="lines+markers",
            line=dict(color=line_color, width=2),
            marker=dict(size=6),
            name=y_label,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="日期",
        yaxis_title=y_label,
        height=height,
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def plot_dual_time_series(
    dates: list,
    values1: list,
    values2: list,
    label1: str = "Series 1",
    label2: str = "Series 2",
    y_label: str = "Value",
    title: str = "双序列对比",
    height: int = 400,
):
    """
    Plotly 双序列对比折线图

    返回:
        plotly Figure
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=values1,
            mode="lines+markers",
            name=label1,
            line=dict(width=2),
            marker=dict(size=5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=values2,
            mode="lines+markers",
            name=label2,
            line=dict(width=2, dash="dash"),
            marker=dict(size=5),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="日期",
        yaxis_title=y_label,
        height=height,
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ============================================
# 趋势分析可视化 (Sen+MK)
# ============================================

def plot_trend_scatter(
    x: list,
    y: list,
    slope: float,
    p_value: float,
    trend: str,
    x_label: str = "时间",
    y_label: str = "NDVI",
    title: str = "Sen + Mann-Kendall 趋势分析",
    height: int = 400,
):
    """
    Plotly 趋势散点图 + 拟合线

    参数:
        x: X轴值列表
        y: Y轴值列表
        slope: Sen 斜率
        p_value: MK 检验 p 值
        trend: 趋势方向 ("increasing"/"decreasing"/"no trend")
        x_label, y_label: 轴标签
        title: 标题
        height: 图表高度

    返回:
        plotly Figure
    """
    import plotly.graph_objects as go

    trend_label = {"increasing": "📈 显著增加", "decreasing": "📉 显著减少", "no trend": "➖ 无显著趋势"}

    fig = go.Figure()

    # 散点
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="观测值",
            marker=dict(size=8, color="#1f77b4", opacity=0.7),
        )
    )

    # 拟合线
    x_arr = np.array(x, dtype=float)
    intercept = np.mean(y) - slope * np.mean(x_arr)
    y_fit = intercept + slope * x_arr
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_fit,
            mode="lines",
            name=f"Sen斜率={slope:.4f}",
            line=dict(color="red", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title=f"{title}<br><sup>Sen斜率={slope:.4f} | p={p_value:.4f} | {trend_label.get(trend, trend)}</sup>",
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=height,
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ============================================
# 变化检测可视化
# ============================================

def render_change_map(
    change_data: np.ndarray,
    title: str = "变化检测结果",
    increase_label: str = "增加",
    decrease_label: str = "减少",
    stable_label: str = "不变",
) -> Image.Image:
    """
    渲染变化检测地图

    参数:
        change_data: 变化数组 (>0=增加, <0=减少, ~0=不变)
        title: 标题
        increase_label, decrease_label, stable_label: 图例标签

    返回:
        PIL.Image
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # 自定义色带: 红=减少, 白/灰=不变, 绿=增加
    cmap = plt.cm.RdYlGn

    vmax = max(abs(np.nanmax(change_data)), abs(np.nanmin(change_data)), 0.01)
    im = ax.imshow(change_data, cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=13)
    ax.axis("off")

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("变化幅度", fontsize=10)

    # 图例
    legend_elements = [
        Patch(facecolor="green", label=increase_label),
        Patch(facecolor="gray", label=stable_label),
        Patch(facecolor="red", label=decrease_label),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return Image.open(buf)


# ============================================
# 土地覆盖分类可视化
# ============================================

def render_classification(
    class_map: np.ndarray,
    class_names: list = None,
    class_colors: list = None,
    title: str = "土地覆盖分类",
) -> Image.Image:
    """
    渲染土地覆盖分类结果

    参数:
        class_map: 2D 分类数组 (值 0~N-1)
        class_names: 类别名称列表
        class_colors: 类别颜色列表 (hex)
        title: 标题

    返回:
        PIL.Image
    """
    if class_names is None:
        class_names = ["背景", "水体", "植被", "裸地", "建设用地", "农田", "矿区"]
    if class_colors is None:
        class_colors = ["#444444", "#0066FF", "#00CC44", "#CCCCCC", "#FF4444", "#FFAA00", "#8B4513"]

    n_classes = len(class_names)

    # 创建自定义色带
    cmap = mcolors.ListedColormap(class_colors[:n_classes])
    bounds = np.arange(-0.5, n_classes, 1)
    norm = mcolors.BoundaryNorm(bounds, n_classes)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(class_map, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title, fontsize=13)
    ax.axis("off")

    # 图例
    legend_elements = [
        Patch(facecolor=class_colors[i], label=class_names[i]) for i in range(n_classes)
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=8,
        ncol=2,
        framealpha=0.8,
    )

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return Image.open(buf)


# ============================================
# 直方图统计
# ============================================

def plot_histogram(
    data: np.ndarray,
    bins: int = 50,
    x_label: str = "Value",
    title: str = "分布直方图",
    color: str = "#1f77b4",
    height: int = 350,
):
    """
    Plotly 直方图

    返回:
        plotly Figure
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    valid_data = data[np.isfinite(data)]
    fig.add_trace(
        go.Histogram(
            x=valid_data.flatten(),
            nbinsx=bins,
            marker_color=color,
            name="分布",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="频数",
        height=height,
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ============================================
# NDVI vs 气候因子 相关性散点图
# ============================================

def plot_correlation_scatter(
    x: list,
    y: list,
    x_label: str = "X",
    y_label: str = "Y",
    title: str = "相关性分析",
    height: int = 400,
):
    """
    Plotly 相关性散点图 (NDVI vs 降水/气温)

    返回:
        plotly Figure
    """
    import plotly.graph_objects as go
    from scipy import stats as scipy_stats

    x_arr = np.array(x)
    y_arr = np.array(y)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]

    r, p = scipy_stats.pearsonr(x_arr, y_arr)

    # 线性拟合
    slope, intercept, _, _, _ = scipy_stats.linregress(x_arr, y_arr)
    x_fit = np.linspace(x_arr.min(), x_arr.max(), 100)
    y_fit = slope * x_fit + intercept

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_arr,
            y=y_arr,
            mode="markers",
            name="散点",
            marker=dict(size=6, color="#1f77b4", opacity=0.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=y_fit,
            mode="lines",
            name=f"r={r:.3f}",
            line=dict(color="red", width=2),
        )
    )

    fig.update_layout(
        title=f"{title}<br><sup>Pearson r={r:.4f} | p={p:.4f}</sup>",
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=height,
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ============================================
# 多级变化检测可视化 (7级分类)
# ============================================

# 7级变化配色方案
MULTILEVEL_COLORS = {
    -3: "#8B0000",  # 深红 — 严重减少
    -2: "#FF4444",  # 中红 — 中度减少
    -1: "#FFB3B3",  # 浅红 — 轻度减少
    0:  "#CCCCCC",  # 灰色 — 稳定
    1:  "#B3FFB3",  # 浅绿 — 轻度增加
    2:  "#44FF44",  # 中绿 — 中度增加
    3:  "#006600",  # 深绿 — 严重增加
}

MULTILEVEL_LABELS = {
    -3: "严重减少",
    -2: "中度减少",
    -1: "轻度减少",
    0:  "稳定",
    1:  "轻度增加",
    2:  "中度增加",
    3:  "严重增加",
}


def render_multilevel_change(
    change_class: np.ndarray,
    title: str = "多级变化检测",
    figsize: tuple = (10, 8),
    dpi: int = 100,
) -> Image.Image:
    """
    渲染7级变化检测地图

    参数:
        change_class: 2D int8数组, 值范围 -3 ~ +3
        title: 标题
        figsize: 图表尺寸
        dpi: 分辨率

    返回:
        PIL.Image
    """
    # 构建颜色映射: -3~+3 共7级
    n_levels = 7
    colors_ordered = [MULTILEVEL_COLORS[i] for i in range(-3, 4)]
    labels_ordered = [MULTILEVEL_LABELS[i] for i in range(-3, 4)]

    cmap = mcolors.ListedColormap(colors_ordered)
    bounds = np.arange(-3.5, 4, 1)  # -3.5, -2.5, ..., 3.5
    norm = mcolors.BoundaryNorm(bounds, n_levels)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(change_class, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    # 图例
    legend_elements = [
        Patch(facecolor=colors_ordered[i], label=labels_ordered[i])
        for i in range(n_levels)
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=9,
        ncol=2,
        framealpha=0.85,
        title="变化级别",
        title_fontsize=10,
    )

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return Image.open(buf)


def render_change_overlay(
    rgb_image: Image.Image,
    change_class: np.ndarray,
    title: str = "变化检测叠加图",
    alpha: float = 0.55,
    figsize: tuple = (10, 8),
    dpi: int = 100,
) -> Image.Image:
    """
    RGB底图 + 半透明变化分类叠加

    参数:
        rgb_image: RGB底图 PIL.Image (需与 change_class 尺寸一致)
        change_class: 2D int8数组, 值范围 -3 ~ +3
        title: 标题
        alpha: 变化叠加层的透明度 (0~1)
        figsize: 图表尺寸
        dpi: 分辨率

    返回:
        PIL.Image
    """
    rgb_array = np.array(rgb_image).astype(np.float32) / 255.0

    # 构建 RGBA 变化叠加层
    h, w = change_class.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)

    # -- 增加区域 (绿色系) --
    for level in [1, 2, 3]:
        mask = change_class == level
        color_hex = MULTILEVEL_COLORS[level].lstrip("#")
        r_val = int(color_hex[0:2], 16) / 255.0
        g_val = int(color_hex[2:4], 16) / 255.0
        b_val = int(color_hex[4:6], 16) / 255.0
        overlay[mask] = [r_val, g_val, b_val, alpha]

    # -- 减少区域 (红色系) --
    for level in [-1, -2, -3]:
        mask = change_class == level
        color_hex = MULTILEVEL_COLORS[level].lstrip("#")
        r_val = int(color_hex[0:2], 16) / 255.0
        g_val = int(color_hex[2:4], 16) / 255.0
        b_val = int(color_hex[4:6], 16) / 255.0
        overlay[mask] = [r_val, g_val, b_val, alpha]

    fig, ax = plt.subplots(figsize=figsize)

    # RGB 底图
    ax.imshow(rgb_array)

    # 叠加变化图层
    ax.imshow(overlay)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    # 图例
    legend_elements = [
        Patch(facecolor=MULTILEVEL_COLORS[3], label="严重增加", alpha=alpha),
        Patch(facecolor=MULTILEVEL_COLORS[2], label="中度增加", alpha=alpha),
        Patch(facecolor=MULTILEVEL_COLORS[1], label="轻度增加", alpha=alpha),
        Patch(facecolor=MULTILEVEL_COLORS[0], label="稳定", alpha=alpha),
        Patch(facecolor=MULTILEVEL_COLORS[-1], label="轻度减少", alpha=alpha),
        Patch(facecolor=MULTILEVEL_COLORS[-2], label="中度减少", alpha=alpha),
        Patch(facecolor=MULTILEVEL_COLORS[-3], label="严重减少", alpha=alpha),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=8,
        ncol=2,
        framealpha=0.85,
        title="变化级别",
        title_fontsize=9,
    )

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return Image.open(buf)


def plot_multilevel_change_stacked_bar(
    level_counts: dict,
    total_pixels: int,
    title: str = "多级变化分布",
    height: int = 400,
):
    """
    Plotly 多级变化水平堆叠条形图

    参数:
        level_counts: {level_id: pixel_count}, 如 {-3: 1500, -2: 300, ...}
        total_pixels: 总有效像素数
        title: 标题
        height: 图表高度

    返回:
        plotly Figure
    """
    import plotly.graph_objects as go

    levels = list(range(-3, 4))
    colors_ordered = [MULTILEVEL_COLORS[l] for l in levels]
    labels_ordered = [MULTILEVEL_LABELS[l] for l in levels]

    counts = [level_counts.get(l, 0) for l in levels]
    percentages = [c / max(total_pixels, 1) * 100 for c in counts]

    # 构建悬停文本
    hover_texts = [
        f"{labels_ordered[i]}<br>{counts[i]:,} 像素<br>{percentages[i]:.2f}%"
        for i in range(len(levels))
    ]

    fig = go.Figure()

    # 负向变化 (左半部分)
    neg_indices = [0, 1, 2]  # levels -3, -2, -1
    for i in neg_indices:
        fig.add_trace(go.Bar(
            y=["变化面积"],
            x=[-percentages[i]],
            name=labels_ordered[i],
            orientation="h",
            marker_color=colors_ordered[i],
            text=hover_texts[i],
            textposition="inside",
            insidetextanchor="middle",
            hoverinfo="text",
        ))

    # 稳定 (中间)
    fig.add_trace(go.Bar(
        y=["变化面积"],
        x=[percentages[3]],
        name=labels_ordered[3],
        orientation="h",
        marker_color=colors_ordered[3],
        text=hover_texts[3],
        textposition="inside",
        insidetextanchor="middle",
        hoverinfo="text",
    ))

    # 正向变化 (右半部分)
    pos_indices = [4, 5, 6]  # levels 1, 2, 3
    for i in pos_indices:
        fig.add_trace(go.Bar(
            y=["变化面积"],
            x=[percentages[i]],
            name=labels_ordered[i],
            orientation="h",
            marker_color=colors_ordered[i],
            text=hover_texts[i],
            textposition="inside",
            insidetextanchor="middle",
            hoverinfo="text",
        ))

    fig.update_layout(
        title=title,
        barmode="relative",
        height=height,
        template="plotly_white",
        xaxis_title="面积占比 (%)",
        xaxis=dict(
            tickvals=[-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100],
            ticktext=["100%", "80%", "60%", "40%", "20%", "0%", "20%", "40%", "60%", "80%", "100%"],
        ),
        margin=dict(l=20, r=20, t=50, b=30),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
        ),
    )

    return fig
