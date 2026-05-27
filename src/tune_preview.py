"""
快速调参预览工具 —— 修改顶部 PARAMS 区域后运行即可预览

用法:
    python src/tune_preview.py

每次修改参数后重新运行，生成的预览图在 frames/_preview/ 目录。
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ═══════════════════════════════════════════════════════════════
#  ▎ 可调参数区 —— 修改这里即可
# ═══════════════════════════════════════════════════════════════

class P:
    # ── 数据选择 ──────────────────────────────────────
    VARIABLE   = "NEE"         # GPP | TER | NEE
    FOREST     = "Forest"      # Broadleaf | Needleleaf | Forest
    SCENARIO   = "ssp585"      # ssp126 | ssp245 | ssp585
    YEAR       = 2060           # 2015–2060

    # ── 图片背景 ──────────────────────────────────────
    FIG_BG     = "white"       # 图形背景色
    AX_BG      = "white"       # 坐标区背景色
    LAND_BG    = "white"       # 陆地填充色 (改纯白)
    OCEAN_BG   = "white"       # 海洋填充色

    # ── 边界扩展 ──────────────────────────────────────
    MAP_PAD_RATIO = 0.06        # 地图边界向外扩展比例 (0=贴边, 0.06=留白6%)

    # ── 图片尺寸 ──────────────────────────────────────
    FIG_W      = 12            # 英寸
    FIG_H      = 7             # 英寸
    DPI        = 150

    # ── 海岸线/国界 ───────────────────────────────────
    COAST_COLOR  = "#888888"
    COAST_WIDTH  = 0.6
    BORDER_COLOR = "#aaaaaa"
    BORDER_WIDTH = 0.4

    # ── 经纬网 ────────────────────────────────────────
    GRID_COLOR   = "#cccccc"
    GRID_WIDTH   = 0.3
    GRID_ALPHA   = 0.35
    GRID_LABEL_SIZE  = 8
    GRID_LABEL_COLOR = "black"
    GRID_STEP    = None         # None=自动; 或手动如 5 (度)

    # ── 色带 ──────────────────────────────────────────
    CBAR_LEFT_OFFSET  = 0.76    # 色带距右边界偏移 (归一化坐标, 越大越靠左)
    CBAR_BOTTOM_RATIO = 0.15     # 色带底部位置 (占图高比例)
    CBAR_WIDTH        = 0.015    # 色带宽度
    CBAR_HEIGHT_RATIO = 0.55     # 色带高度 (占图高比例)
    CBAR_TICK_SIZE    = 7
    CBAR_UNIT_SIZE    = 7.5
    CBAR_UNIT_COLOR   = "#333333"
    CBAR_UNIT_OFFSET  = 1.04     # 单位文字在色带顶部上方的偏移

    # ── 比例尺 ────────────────────────────────────────
    SCALE_STYLE    = "blocks"   # "classic" 经典黑线 | "blocks" 黑白交替块 | "line" 极简线
    SCALE_KM       = 500        # 比例尺长度 (公里)
    SCALE_X_RATIO  = 0.05       # 距左边界比例
    SCALE_Y_RATIO  = 0.07       # 距下边界比例

    # ── 指北针 ────────────────────────────────────────
    NORTH_STYLE    = "compass"  # "simple" 简单箭头 | "compass" 四角罗盘 | "circle" 圆底三角
    NORTH_X_RATIO  = 0.06       # 距右边界比例
    NORTH_Y_RATIO  = 0.16       # 距上边界比例

    # ── 标题 ──────────────────────────────────────────
    TITLE_X        = 0.02        # 标题距左边界比例
    TITLE_Y        = 0.96        # 标题距下边界比例
    TITLE_SIZE     = 14
    TITLE_COLOR    = "#222222"
    TITLE_BOX_ALPHA = 0.85

    # ── 配色方案 ──────────────────────────────────────
    # 可在此覆盖默认色带
    CMAP_OVERRIDE  = None        # None=用默认; 或 "YlGn" / "Oranges" / "viridis" 等
    FORCE_VMIN     = None        # None=自动; 或数值
    FORCE_VMAX     = None        # None=自动; 或数值


# ═══════════════════════════════════════════════════════════════
#  ▎ 渲染逻辑 (无需修改)
# ═══════════════════════════════════════════════════════════════

def _setup_font():
    """统一字体: Times New Roman, 黑色"""
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "SimSun", "DejaVu Serif"]
    plt.rcParams["text.color"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"
    plt.rcParams["axes.unicode_minus"] = False
_setup_font()

SCENARIO_SHORT = {"ssp126": "SSP126", "ssp245": "SSP245", "ssp585": "SSP585"}
import warnings
warnings.filterwarnings("ignore")

# 色带
from matplotlib.colors import LinearSegmentedColormap
DEFAULT_CMAPS = {
    "GPP": "YlGn",
    "TER": "Oranges",
    "NEE": LinearSegmentedColormap.from_list("NEP", ["#FFFACD", "#9ACD32", "#006400"]),
}
VARIABLE_UNITS = {"GPP": "GPP (gC/m2/yr)", "TER": "TER (gC/m2/yr)", "NEE": "NEE (gC/m2/yr)"}


def _add_scale_bar(ax, extent, style, length_km, x_ratio, y_ratio):
    """绘制比例尺。style: classic | blocks | line"""
    mid_lat = (extent[2] + extent[3]) / 2
    deg_per_km = 1.0 / (111.32 * np.cos(np.radians(mid_lat)))
    len_deg = length_km * deg_per_km
    x0 = extent[0] + (extent[1] - extent[0]) * x_ratio
    y0 = extent[2] + (extent[3] - extent[2]) * y_ratio
    h = 0.10
    fs = 8
    bar_h = h * 2.5

    if style == "blocks":
        # 规范比例尺: 4段 + 细分刻度 + 多级标注
        n_seg = 4
        seg_w = len_deg / n_seg
        colors = ["black", "white", "black", "white"]
        for i in range(n_seg):
            rect = plt.Rectangle((x0 + i * seg_w, y0 - bar_h), seg_w, bar_h,
                                 facecolor=colors[i], edgecolor="#333333", linewidth=0.5,
                                 transform=ccrs.PlateCarree(), zorder=10)
            ax.add_patch(rect)
        # 上方细分刻度线
        n_sub = 8
        sub_w = len_deg / n_sub
        for i in range(n_sub + 1):
            tick_h = bar_h * 0.6 if i % 2 == 0 else bar_h * 0.35
            ax.plot([x0 + i * sub_w, x0 + i * sub_w], [y0, y0 - tick_h],
                    color="#333333", linewidth=0.7,
                    transform=ccrs.PlateCarree(), zorder=11)
        # 下方标注 + 单位紧跟最后数字
        for i in range(n_seg):
            val = i * (length_km // n_seg)
            ax.text(x0 + i * seg_w, y0 - bar_h - h * 2.0, str(val),
                    ha="center", va="top", fontsize=fs,
                    transform=ccrs.PlateCarree(), zorder=10)
        # 最后一个数字后紧跟 km
        last_val = n_seg * (length_km // n_seg)
        ax.text(x0 + n_seg * seg_w, y0 - bar_h - h * 2.0, f"{last_val} km",
                ha="center", va="top", fontsize=fs,
                transform=ccrs.PlateCarree(), zorder=10)

    elif style == "line":
        ax.plot([x0, x0 + len_deg], [y0, y0], color="#333333", linewidth=3,
                solid_capstyle="butt", transform=ccrs.PlateCarree(), zorder=10)
        tri_h = h * 2; tri_w = h * 1.5
        for bx in [x0, x0 + len_deg]:
            ax.fill([bx - tri_w, bx, bx + tri_w], [y0 - tri_h, y0, y0 - tri_h],
                    color="#333333", transform=ccrs.PlateCarree(), zorder=10)
        ax.text(x0 + len_deg / 2, y0 - h * 3.5, f"{length_km} km",
                ha="center", va="top", fontsize=fs, fontweight="bold", color="#333333",
                transform=ccrs.PlateCarree(), zorder=10)

    else:  # classic
        ax.plot([x0, x0 + len_deg], [y0, y0], color="black", linewidth=2.5,
                transform=ccrs.PlateCarree(), zorder=10)
        for bx in [x0, x0 + len_deg]:
            ax.plot([bx, bx], [y0 - h, y0 + h], color="black", linewidth=2,
                    transform=ccrs.PlateCarree(), zorder=10)
        ax.text(x0 + len_deg / 2, y0 - h * 2.5, f"{length_km} km",
                ha="center", va="top", fontsize=fs, fontweight="bold",
                transform=ccrs.PlateCarree(), zorder=10)


def _add_north_arrow(ax, extent, style, x_ratio, y_ratio):
    """绘制指北针。style: simple | compass | circle"""
    cx = extent[1] - (extent[1] - extent[0]) * x_ratio
    cy = extent[3] - (extent[3] - extent[2]) * y_ratio
    r = 1.8  # 基础半径(度)

    if style == "compass":
        # 四角罗盘：菱形 + 半填充
        diamond = np.array([[cx, cy + r], [cx + r * 0.5, cy], [cx, cy - r * 0.6],
                            [cx - r * 0.5, cy]])
        ax.fill(diamond[:, 0], diamond[:, 1], facecolor="white", edgecolor="#333333",
                linewidth=1.2, transform=ccrs.PlateCarree(), zorder=10)
        # 上半填充(北)
        ax.fill([cx - r * 0.45, cx, cx + r * 0.45], [cy, cy + r * 0.85, cy],
                facecolor="#333333", edgecolor="none",
                transform=ccrs.PlateCarree(), zorder=11)
        ax.text(cx, cy + r * 1.35, "N", ha="center", va="center",
                fontsize=10, fontweight="bold", color="#333333",
                transform=ccrs.PlateCarree(), zorder=12)

    elif style == "circle":
        # 圆底三角
        circle = plt.Circle((cx, cy), r * 0.85, facecolor="white", edgecolor="#555555",
                            linewidth=1, transform=ccrs.PlateCarree(), zorder=10)
        ax.add_patch(circle)
        tri = np.array([[cx, cy + r], [cx + r * 0.35, cy - r * 0.3],
                        [cx - r * 0.35, cy - r * 0.3]])
        ax.fill(tri[:, 0], tri[:, 1], facecolor="#cc3333", edgecolor="none",
                transform=ccrs.PlateCarree(), zorder=11)
        ax.text(cx, cy - r * 0.85, "N", ha="center", va="top",
                fontsize=9, fontweight="bold", color="#555555",
                transform=ccrs.PlateCarree(), zorder=12)

    else:  # simple
        ax.annotate("N", xy=(cx, cy + r), xytext=(cx, cy),
                    arrowprops=dict(facecolor="#333333", width=2.5, headwidth=8, headlength=8),
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="#333333", transform=ccrs.PlateCarree(), zorder=10)


def _add_colorbar(fig, ax, mesh, unit_label):
    bbox = ax.get_position()
    cax = fig.add_axes([
        bbox.x1 - P.CBAR_LEFT_OFFSET,
        bbox.y0 + bbox.height * P.CBAR_BOTTOM_RATIO,
        P.CBAR_WIDTH,
        bbox.height * P.CBAR_HEIGHT_RATIO,
    ])
    cbar = plt.colorbar(mesh, cax=cax, orientation="vertical")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=P.CBAR_TICK_SIZE, length=2, pad=2)
    cax.text(0.0, P.CBAR_UNIT_OFFSET, unit_label, transform=cax.transAxes,
             ha="left", va="bottom", fontsize=P.CBAR_UNIT_SIZE,
             fontweight="bold", color=P.CBAR_UNIT_COLOR)


def render_preview():
    from core import DataLoader, load_and_average, load_forest_average
    from compute_color_limits import load_color_limits
    from config import INVENTORY_CSV

    loader = DataLoader(INVENTORY_CSV)
    color_limits = load_color_limits()
    lon, lat = loader.lon_lat_grid()

    # 加载数据
    key = f"{P.VARIABLE}_{P.FOREST}"
    clim = color_limits.get(key, {"vmin": 0, "vmax": 100})
    vmin = P.FORCE_VMIN if P.FORCE_VMIN is not None else clim["vmin"]
    vmax = P.FORCE_VMAX if P.FORCE_VMAX is not None else clim["vmax"]

    if P.FOREST == "Forest":
        data = load_forest_average(loader, P.SCENARIO, P.VARIABLE, P.YEAR)
    else:
        data = load_and_average(loader, P.SCENARIO, P.FOREST, P.VARIABLE, P.YEAR)

    print(f"数据: {key} | 年份={P.YEAR}")
    print(f"范围: [{np.nanmin(data):.1f}, {np.nanmax(data):.1f}]")
    print(f"色带: vmin={vmin} vmax={vmax}")

    # 色带
    cmap = P.CMAP_OVERRIDE if P.CMAP_OVERRIDE else DEFAULT_CMAPS.get(P.VARIABLE, "viridis")

    # 创建图形
    fig = plt.figure(figsize=(P.FIG_W, P.FIG_H), dpi=P.DPI, facecolor=P.FIG_BG)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_facecolor(P.AX_BG)

    extent_raw = [lon.min(), lon.max(), lat.min(), lat.max()]
    # 边界扩展
    lon_span = extent_raw[1] - extent_raw[0]
    lat_span = extent_raw[3] - extent_raw[2]
    extent = [
        extent_raw[0] - lon_span * P.MAP_PAD_RATIO,
        extent_raw[1] + lon_span * P.MAP_PAD_RATIO,
        extent_raw[2] - lat_span * P.MAP_PAD_RATIO,
        extent_raw[3] + lat_span * P.MAP_PAD_RATIO,
    ]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # 底图
    ax.add_feature(cfeature.LAND, facecolor=P.LAND_BG, zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor=P.OCEAN_BG, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=P.COAST_WIDTH, edgecolor=P.COAST_COLOR, zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=P.BORDER_WIDTH, edgecolor=P.BORDER_COLOR, zorder=2)

    # 经纬网 (加密 + 四角标注)
    lon_ticks = _dense_ticks(extent[0], extent[1], P.GRID_STEP)
    lat_ticks = _dense_ticks(extent[2], extent[3], P.GRID_STEP)
    gl = ax.gridlines(draw_labels=True, linewidth=P.GRID_WIDTH,
                      color=P.GRID_COLOR, alpha=P.GRID_ALPHA, linestyle="--",
                      xlocs=lon_ticks, ylocs=lat_ticks)
    gl.top_labels = True
    gl.right_labels = True
    gl.left_labels = True
    gl.bottom_labels = True
    gl.xlabel_style = {"size": P.GRID_LABEL_SIZE, "color": P.GRID_LABEL_COLOR}
    gl.ylabel_style = {"size": P.GRID_LABEL_SIZE, "color": P.GRID_LABEL_COLOR}

    # 栅格 (零值区域 → 透明白底)
    data_masked = np.where(np.abs(data) < 0.001, np.nan, data)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.cm.get_cmap(cmap).copy()
    cmap_obj.set_bad("white", alpha=0)  # NaN → 纯透明
    mesh = ax.pcolormesh(lon, lat, data_masked, cmap=cmap_obj, norm=norm,
                         transform=ccrs.PlateCarree(), rasterized=True, shading="auto")

    # 比例尺
    _add_scale_bar(ax, extent, P.SCALE_STYLE, P.SCALE_KM, P.SCALE_X_RATIO, P.SCALE_Y_RATIO)

    # 指北针
    _add_north_arrow(ax, extent, P.NORTH_STYLE, P.NORTH_X_RATIO, P.NORTH_Y_RATIO)

    # 色带
    unit_label = VARIABLE_UNITS.get(P.VARIABLE, "")
    _add_colorbar(fig, ax, mesh, unit_label)

    # 标题
    scenario_short = SCENARIO_SHORT.get(P.SCENARIO, P.SCENARIO.upper())
    title_text = f"{scenario_short}-{P.VARIABLE}-{P.YEAR}"
    ax.text(P.TITLE_X, P.TITLE_Y, title_text,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=P.TITLE_SIZE, fontweight="bold", color=P.TITLE_COLOR,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=P.TITLE_BOX_ALPHA))

    # 保存
    out_dir = Path(r"d:\Programs\FigureAgent\frames\_preview")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"preview_{P.VARIABLE}_{P.FOREST}_{P.SCENARIO}_{P.YEAR}.png"
    fig.savefig(out_path, dpi=P.DPI, bbox_inches="tight",
                facecolor=P.FIG_BG, edgecolor="none")
    plt.close(fig)

    print(f"\n预览图已保存: {out_path}")
    print("=" * 60)
    print("当前参数摘要:")
    print(f"  背景: FIG={P.FIG_BG} AX={P.AX_BG} LAND={P.LAND_BG} OCEAN={P.OCEAN_BG}")
    print(f"  边界: PAD={P.MAP_PAD_RATIO} ({lon_span*P.MAP_PAD_RATIO:.1f}°×{lat_span*P.MAP_PAD_RATIO:.1f}°)")
    print(f"  经纬网: STEP={P.GRID_STEP or 'auto'} COLOR={P.GRID_LABEL_COLOR}")
    print(f"  色带: LEFT_OFFSET={P.CBAR_LEFT_OFFSET} W={P.CBAR_WIDTH} H_RATIO={P.CBAR_HEIGHT_RATIO}")
    print(f"  比例尺: STYLE={P.SCALE_STYLE} {P.SCALE_KM}km X={P.SCALE_X_RATIO} Y={P.SCALE_Y_RATIO}")
    print(f"  指北针: STYLE={P.NORTH_STYLE} X={P.NORTH_X_RATIO} Y={P.NORTH_Y_RATIO}")
    print(f"  标题: ({P.TITLE_X}, {P.TITLE_Y}) SIZE={P.TITLE_SIZE}")


def _dense_ticks(vmin, vmax, step_override=None):
    """生成密集刻度，确保起止点在刻度列表中。"""
    if step_override:
        step = step_override
    else:
        span = vmax - vmin
        step = 1 if span <= 12 else 5 if span <= 40 else 10
    start = np.ceil(vmin / step) * step
    end = np.floor(vmax / step) * step
    return list(np.arange(start, end + step * 0.5, step))


def _nice_ticks(vmin, vmax):
    span = vmax - vmin
    step = 1 if span <= 10 else 5 if span <= 30 else 10 if span <= 60 else \
           15 if span <= 120 else 30 if span <= 200 else 60
    start = np.ceil(vmin / step) * step
    end = np.floor(vmax / step) * step
    return list(np.arange(start, end + step * 0.5, step))


if __name__ == "__main__":
    render_preview()
