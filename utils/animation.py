"""
时序动画生成模块
================
生成遥感指数年际/季节变化 GIF 动画

支持:
  - NDVI 时序动画 (植被变化)
  - MNDWI 水体变化动画
  - NDSI 雪盖消长动画
  - 多指数并列动画

输出: GIF / MP4 (需 ffmpeg)

依赖: numpy, matplotlib, imageio, PIL
"""

import numpy as np
import warnings
from typing import Optional, List, Tuple, Dict
from io import BytesIO
from pathlib import Path

warnings.filterwarnings("ignore")

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
# 动画生成核心
# ============================================================

def create_timeseries_animation(
    data_stack: np.ndarray,
    dates: List[str],
    output_path: Optional[str] = None,
    cmap: str = "RdYlGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "NDVI 时序变化",
    label: str = "NDVI",
    fps: int = 2,
    dpi: int = 100,
    figsize: Tuple[int, int] = (8, 6),
    add_colorbar: bool = True,
    show_date: bool = True,
    show_stat: bool = True,
) -> Optional[bytes]:
    """
    生成单指数时序动画 (GIF)

    参数:
        data_stack: (H, W, T) 指数时序立方体
        dates: 日期标签列表 (T,)
        output_path: 输出路径, None=返回 bytes
        cmap: Matplotlib colormap
        vmin/vmax: 颜色范围, None=自动
        title: 动画标题
        label: 色条标签
        fps: 帧率
        dpi: 分辨率
        figsize: 图尺寸 (英寸)
        add_colorbar: 是否加色条
        show_date: 是否显示日期
        show_stat: 是否显示统计

    返回:
        GIF bytes (output_path=None) 或 None (output_path指定)
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from PIL import Image
    import io

    data_stack = np.asarray(data_stack, dtype=np.float64)
    if data_stack.ndim != 3:
        raise ValueError(f"data_stack 必须是 3D (H,W,T), 实际: {data_stack.ndim}D")

    H, W, T = data_stack.shape

    # 自动范围
    if vmin is None:
        vmin = float(np.nanpercentile(data_stack, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(data_stack, 98))

    # 预计算统计
    mean_series = np.array([np.nanmean(data_stack[:, :, t]) for t in range(T)])

    frames = []
    for t in range(T):
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(data_stack[:, :, t], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis("off")

        # 标题
        ax.set_title(title, fontsize=14, fontweight="bold")

        # 日期
        if show_date and t < len(dates):
            ax.text(0.02, 0.98, dates[t], transform=ax.transAxes,
                    fontsize=11, color="white", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6))

        # 统计
        if show_stat:
            mean_val = mean_series[t]
            ax.text(0.98, 0.02, f"Mean: {mean_val:.4f}",
                    transform=ax.transAxes, fontsize=9, color="white", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5))

        # 色条
        if add_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04, shrink=0.8)
            cbar.set_label(label, fontsize=10)

        plt.tight_layout()

        # 渲染为帧
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(np.array(Image.open(buf)))

    # 合成 GIF
    try:
        import imageio
        gif_buf = BytesIO()
        imageio.mimsave(gif_buf, frames, format="GIF", fps=fps, loop=0)
        gif_bytes = gif_buf.getvalue()

        if output_path:
            with open(output_path, "wb") as f:
                f.write(gif_bytes)
            return None
        else:
            return gif_bytes

    except ImportError:
        # 降级: 用 PIL 合成
        if len(frames) > 0:
            pil_frames = [Image.fromarray(f) for f in frames]
            gif_buf = BytesIO()
            pil_frames[0].save(
                gif_buf, format="GIF", save_all=True,
                append_images=pil_frames[1:],
                duration=int(1000 / fps), loop=0,
            )
            return gif_buf.getvalue()
        return None


def create_multi_index_animation(
    data_stacks: Dict[str, np.ndarray],
    dates: List[str],
    output_path: Optional[str] = None,
    configs: Optional[Dict[str, Dict]] = None,
    fps: int = 2,
    dpi: int = 100,
    title: str = "多指数时序对比",
) -> Optional[bytes]:
    """
    多指数并列动画

    参数:
        data_stacks: {"NDVI": (H,W,T), "MNDWI": (H,W,T), ...}
        dates: 日期标签
        configs: {name: {cmap, vmin, vmax, label}}
        fps/dpi/title

    返回:
        GIF bytes
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    n_indices = len(data_stacks)
    if n_indices == 0:
        return None

    first_name = list(data_stacks.keys())[0]
    first_data = np.asarray(list(data_stacks.values())[0], dtype=np.float64)
    T = first_data.shape[-1]

    # 默认配置
    default_configs = {
        "NDVI": {"cmap": "RdYlGn", "vmin": -0.2, "vmax": 0.9, "label": "NDVI"},
        "MNDWI": {"cmap": "Blues", "vmin": -0.5, "vmax": 0.5, "label": "MNDWI"},
        "NDSI": {"cmap": "Blues_r", "vmin": -0.3, "vmax": 0.9, "label": "NDSI"},
        "NDDI": {"cmap": "YlOrRd", "vmin": -0.5, "vmax": 1.0, "label": "NDDI"},
        "VCI": {"cmap": "YlOrRd_r", "vmin": 0, "vmax": 100, "label": "VCI"},
    }
    if configs is None:
        configs = {name: default_configs.get(name, {"cmap": "viridis", "vmin": None, "vmax": None, "label": name})
                   for name in data_stacks}

    frames = []
    for t in range(T):
        fig, axes = plt.subplots(1, n_indices, figsize=(4.5 * n_indices, 5))
        if n_indices == 1:
            axes = [axes]

        for ax_i, (name, data) in enumerate(data_stacks.items()):
            arr = np.asarray(data, dtype=np.float64)
            cfg = configs.get(name, {})
            vmin = cfg.get("vmin")
            vmax = cfg.get("vmax")
            if vmin is None:
                vmin = float(np.nanpercentile(arr, 2))
            if vmax is None:
                vmax = float(np.nanpercentile(arr, 98))

            im = axes[ax_i].imshow(arr[:, :, t], cmap=cfg.get("cmap", "viridis"),
                                   vmin=vmin, vmax=vmax)
            axes[ax_i].set_title(f"{name}", fontsize=12, fontweight="bold")
            axes[ax_i].axis("off")
            plt.colorbar(im, ax=axes[ax_i], fraction=0.04, pad=0.04,
                        label=cfg.get("label", name))

        if t < len(dates):
            fig.suptitle(f"{title}\n{dates[t]}", fontsize=13, y=1.02)

        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(np.array(Image.open(buf)))

    # 合成
    try:
        import imageio
        gif_buf = BytesIO()
        imageio.mimsave(gif_buf, frames, format="GIF", fps=fps, loop=0)
        return gif_buf.getvalue()
    except ImportError:
        pil_frames = [Image.fromarray(f) for f in frames]
        gif_buf = BytesIO()
        pil_frames[0].save(gif_buf, format="GIF", save_all=True,
                          append_images=pil_frames[1:],
                          duration=int(1000 / fps), loop=0)
        return gif_buf.getvalue()


def create_trend_animation(
    timeseries: np.ndarray,
    dates: List[str],
    title: str = "NDVI 趋势变化",
    ylabel: str = "NDVI",
    highlight_last: int = 0,
) -> Optional[bytes]:
    """
    创建时序趋势折线图动画

    适用于展示 NDVI 随时间变化的动态曲线

    参数:
        timeseries: (T,) 时序均值
        dates: 日期标签
        title/ylabel
        highlight_last: 高亮最后 N 帧

    返回:
        GIF bytes
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    timeseries = np.asarray(timeseries, dtype=np.float64)
    T = len(timeseries)

    frames = []
    for t in range(1, T + 1):
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(t)
        y = timeseries[:t]

        ax.plot(x, y, "o-", color="#2ecc71", linewidth=2, markersize=6)
        ax.fill_between(x, y, alpha=0.15, color="#2ecc71")

        # 高亮
        if highlight_last > 0:
            hl_start = max(0, t - highlight_last)
            ax.plot(x[hl_start:], y[hl_start:], "o-", color="#e74c3c",
                    linewidth=2.5, markersize=8)

        ax.set_ylim(timeseries.min() - 0.05, timeseries.max() + 0.05)
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_xlabel("时相序号", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{title} ({dates[t-1] if t <= len(dates) else ''})", fontsize=13)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(np.array(Image.open(buf)))

    pil_frames = [Image.fromarray(f) for f in frames]
    gif_buf = BytesIO()
    pil_frames[0].save(gif_buf, format="GIF", save_all=True,
                      append_images=pil_frames[1:],
                      duration=500, loop=0)
    return gif_buf.getvalue()
