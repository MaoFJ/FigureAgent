"""
Phase 4 v3: 地图可视化 — 与 tune_preview 参数同步
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

from config import (
    CMAPS, VARIABLE_UNITS,
    SCENARIO_SHORT, FIGURE_DPI, FIGURE_WIDTH, FIGURE_HEIGHT,
    BG_COLOR, MAP_PAD_RATIO,
)

# ── 字体: Times New Roman, 黑色 ───────────────────────

def _setup_font():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "SimSun", "DejaVu Serif"]
    plt.rcParams["text.color"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"
    plt.rcParams["axes.unicode_minus"] = False
_setup_font()
warnings.filterwarnings("ignore", category=UserWarning, module="cartopy")


# ── 比例尺 (blocks 规范风格) ──────────────────────────

def _add_scale_bar(ax, extent, length_km=500):
    mid_lat = (extent[2] + extent[3]) / 2
    deg_per_km = 1.0 / (111.32 * np.cos(np.radians(mid_lat)))
    len_deg = length_km * deg_per_km
    x0 = extent[0] + (extent[1] - extent[0]) * 0.05
    y0 = extent[2] + (extent[3] - extent[2]) * 0.07
    h = 0.10; fs = 8; bar_h = h * 2.5

    n_seg = 4; seg_w = len_deg / n_seg
    colors = ["black", "white", "black", "white"]
    for i in range(n_seg):
        rect = plt.Rectangle((x0 + i * seg_w, y0 - bar_h), seg_w, bar_h,
                             facecolor=colors[i], edgecolor="#333333", linewidth=0.5,
                             transform=ccrs.PlateCarree(), zorder=10)
        ax.add_patch(rect)
    n_sub = 8; sub_w = len_deg / n_sub
    for i in range(n_sub + 1):
        tick_h = bar_h * 0.6 if i % 2 == 0 else bar_h * 0.35
        ax.plot([x0 + i * sub_w, x0 + i * sub_w], [y0, y0 - tick_h],
                color="#333333", linewidth=0.7,
                transform=ccrs.PlateCarree(), zorder=11)
    for i in range(n_seg):
        val = i * (length_km // n_seg)
        ax.text(x0 + i * seg_w, y0 - bar_h - h * 2.0, str(val),
                ha="center", va="top", fontsize=fs,
                transform=ccrs.PlateCarree(), zorder=10)
    last_val = n_seg * (length_km // n_seg)
    ax.text(x0 + n_seg * seg_w, y0 - bar_h - h * 2.0, f"{last_val} km",
            ha="center", va="top", fontsize=fs,
            transform=ccrs.PlateCarree(), zorder=10)


# ── 指北针 (compass 罗盘) ─────────────────────────────

def _add_north_arrow(ax, extent):
    cx = extent[1] - (extent[1] - extent[0]) * 0.06
    cy = extent[3] - (extent[3] - extent[2]) * 0.16
    r = 1.8
    diamond = np.array([[cx, cy + r], [cx + r * 0.5, cy], [cx, cy - r * 0.6],
                        [cx - r * 0.5, cy]])
    ax.fill(diamond[:, 0], diamond[:, 1], facecolor="white", edgecolor="#333333",
            linewidth=1.2, transform=ccrs.PlateCarree(), zorder=10)
    ax.fill([cx - r * 0.45, cx, cx + r * 0.45], [cy, cy + r * 0.85, cy],
            facecolor="#333333", edgecolor="none",
            transform=ccrs.PlateCarree(), zorder=11)
    ax.text(cx, cy + r * 1.35, "N", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#333333",
            transform=ccrs.PlateCarree(), zorder=12)


# ── 竖版色带 (左对齐标签) ─────────────────────────────

def _add_vertical_colorbar(fig, ax, mesh, unit_label):
    bbox = ax.get_position()
    cax = fig.add_axes([
        bbox.x1 - 0.76,
        bbox.y0 + bbox.height * 0.15,
        0.015,
        bbox.height * 0.55,
    ])
    cbar = plt.colorbar(mesh, cax=cax, orientation="vertical")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7, length=2, pad=2)
    cax.text(0.0, 1.04, unit_label, transform=cax.transAxes,
             ha="left", va="bottom", fontsize=7.5, fontweight="bold",
             color="#333333")
    return cbar


# ── 主绘图函数 ────────────────────────────────────────

def plot_single_map(
    data: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    variable: str,
    forest_type: str,
    scenario: str,
    year: int,
    model_label: str,
    vmin: float,
    vmax: float,
    save_path: Path | None = None,
) -> plt.Figure:

    cmap = CMAPS.get(variable, "viridis")

    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=FIGURE_DPI,
                     facecolor=BG_COLOR)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_facecolor(BG_COLOR)

    # 边界扩展
    extent_raw = [lon.min(), lon.max(), lat.min(), lat.max()]
    lon_span = extent_raw[1] - extent_raw[0]
    lat_span = extent_raw[3] - extent_raw[2]
    extent = [
        extent_raw[0] - lon_span * MAP_PAD_RATIO,
        extent_raw[1] + lon_span * MAP_PAD_RATIO,
        extent_raw[2] - lat_span * MAP_PAD_RATIO,
        extent_raw[3] + lat_span * MAP_PAD_RATIO,
    ]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # 底图
    ax.add_feature(cfeature.LAND, facecolor="white", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor=BG_COLOR, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#888888", zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#aaaaaa", zorder=2)

    # 经纬网 (加密 + 四边标注)
    lon_ticks = _dense_ticks(extent[0], extent[1])
    lat_ticks = _dense_ticks(extent[2], extent[3])
    gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                      color="#cccccc", alpha=0.35, linestyle="--",
                      xlocs=lon_ticks, ylocs=lat_ticks)
    gl.top_labels = True; gl.right_labels = True
    gl.left_labels = True; gl.bottom_labels = True
    gl.xlabel_style = {"size": 8, "color": "black"}
    gl.ylabel_style = {"size": 8, "color": "black"}

    # 栅格 (零值透明)
    data_masked = np.where(np.abs(data) < 0.001, np.nan, data)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.cm.get_cmap(cmap).copy()
    cmap_obj.set_bad("white", alpha=0)
    mesh = ax.pcolormesh(lon, lat, data_masked, cmap=cmap_obj, norm=norm,
                         transform=ccrs.PlateCarree(), rasterized=True, shading="auto")

    # 比例尺 + 指北针
    _add_scale_bar(ax, extent)
    _add_north_arrow(ax, extent)

    # 色带
    unit_label = VARIABLE_UNITS.get(variable, "")
    _add_vertical_colorbar(fig, ax, mesh, unit_label)

    # 标题
    scenario_short = SCENARIO_SHORT.get(scenario, scenario.upper())
    title_text = f"{scenario_short}-{variable}-{year}"
    ax.text(0.02, 0.96, title_text,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=14, fontweight="bold", color="#222222",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.85))

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight",
                     facecolor=BG_COLOR, edgecolor="none")
        plt.close(fig)

    return fig


def _dense_ticks(vmin, vmax):
    span = vmax - vmin
    step = 1 if span <= 12 else 5 if span <= 40 else 10
    start = np.ceil(vmin / step) * step
    end = np.floor(vmax / step) * step
    return list(np.arange(start, end + step * 0.5, step))
