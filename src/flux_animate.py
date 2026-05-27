"""
通量站点 GPP / TER / NEE 时间序列 + 散点动态生长图
数据: data/fluxdata.csv (Obs=观测值圆点, Ass=数据同化线条)
输出: 首先生成最后一帧 preview，确认后批量渲染帧 + 合成 GIF
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

# ── 路径 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FLUXDATA_PATH = PROJECT_ROOT / "data" / "fluxdata_2011-2025.csv"
FRAMES_DIR = PROJECT_ROOT / "frames" / "flux_ts"
OUTPUT_DIR = PROJECT_ROOT / "output" / "flux_ts"

# ── 样式常量 ──────────────────────────────────────────
FIG_DPI = 300
FIG_WIDTH = 13
FIG_HEIGHT = 9
LINE_WIDTH = 1.0
DOT_SIZE = 4.0
FRAME_STEP = 3          # 每隔 N 天渲染一帧（被 --duration 覆盖时忽略）
TARGET_DURATION = 10    # 目标 GIF 播放时长（秒）
GIF_DURATION = 0.12     # 每帧时长（秒）
GIF_SCALE = 0.5         # GIF 缩放比例
XAXIS_MODE = "fixed"    # "fixed"=全程固定 | "auto"=X轴随帧扩展

# ── 变量配置（在此处快速调整范围）────────────────────
#   range: (ymin, ymax) — 时序Y轴 & 散点XY轴 统一
#   abs_neg: True 则对原始负值取绝对值（如TER）
VAR_COLORS = {
    "GPP": "#2E7D32",
    "TER": "#E65100",
    "NEE": "#1565C0",
}

VAR_COLS = {
    "GPP": ("GPPObs", "GPPAss"),
    "TER": ("TERObs", "TERAss"),
    "NEE": ("NEEObs", "NEPAss"),
}

VAR_LABELS = {"GPP": "GPP", "TER": "TER", "NEE": "NEE"}
VAR_UNITS = "gC m$^{-2}$ d$^{-1}$"

# 统一定义范围: 时序Y轴 = 散点XY轴
VAR_RANGES = {
    "GPP":  (0, 10),
    "TER":  (1, 4),
    "NEE":  (-2, 5),
}

# TER 原始数据为负值，需取绝对值
VAR_ABS_NEG = {"GPP": False, "TER": True, "NEE": False}


# ═══════════════════════════════════════════════════════════════
#  全局样式
# ═══════════════════════════════════════════════════════════════

def _setup_style():
    """Times New Roman，全部黑色。"""
    import matplotlib.font_manager as fm
    fm._load_fontmanager(try_read_cache=False)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["text.color"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "stix"


# ═══════════════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════════════

def load_flux_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # TER 取绝对值
    for var, abs_neg in VAR_ABS_NEG.items():
        if abs_neg:
            for col in VAR_COLS[var]:
                df[col] = df[col].abs()

    return df


# ═══════════════════════════════════════════════════════════════
#  单帧渲染 — 3×2 面板
# ═══════════════════════════════════════════════════════════════

def render_frame(
    df: pd.DataFrame,
    idx: int,
    save_path: Path,
    y_limits: dict,
    scatter_limits: dict,
    xaxis_mode: str = "fixed",
) -> None:
    """
    渲染第 idx 行为止的数据。

    左列: GPP / TER / NEE 时间序列（Obs 圆点 + Ass 线条）
    右列: Ass vs Obs 散点图（动态累积）
    xaxis_mode: "fixed" 全程固定 X 轴 | "auto" X 轴随帧扩展
    """
    _setup_style()

    subset = df.iloc[: idx + 1]
    current_date = subset["Date"].iloc[-1]
    date_min = df["Date"].min()
    date_max = df["Date"].max()

    fig, axes = plt.subplots(
        3, 2, figsize=(FIG_WIDTH, FIG_HEIGHT),
        dpi=FIG_DPI,
        gridspec_kw={"width_ratios": [2.5, 1]},
    )
    fig.patch.set_facecolor("white")

    variables = ["GPP", "TER", "NEE"]
    year_starts = pd.date_range(date_min, date_max, freq="YS")

    # 计算统一的 X 轴右边界
    if xaxis_mode == "auto":
        xlim_right = current_date + pd.Timedelta(days=30)
    else:
        xlim_right = date_max + pd.Timedelta(days=60)

    # 根据当前 X 轴跨度确定刻度间隔
    span_days = (xlim_right - date_min).days
    if span_days < 365:
        interval = 1       # 月
    elif span_days < 730:
        interval = 2       # 双月
    elif span_days < 1460:
        interval = 3       # 季
    elif span_days < 2190:
        interval = 4
    elif span_days < 3285:
        interval = 6       # 半年
    else:
        interval = 12      # 年

    for i, var in enumerate(variables):
        obs_col, ass_col = VAR_COLS[var]
        color = VAR_COLORS[var]

        # ── 左列：时间序列 ────────────────────────
        ax_ts = axes[i, 0]

        # 观测值：小圆点
        ax_ts.scatter(
            subset["Date"], subset[obs_col],
            color=color, s=DOT_SIZE, linewidths=0, zorder=3,
        )

        # 同化值：线条
        ax_ts.plot(
            subset["Date"], subset[ass_col],
            color=color, linewidth=LINE_WIDTH, zorder=2,
        )

        # Y 轴范围与标签
        ymin, ymax = y_limits[var]
        ax_ts.set_ylim(ymin, ymax)
        ax_ts.set_ylabel(f"{VAR_LABELS[var]} ({VAR_UNITS})", fontsize=9,
                          color="black")

        # 水平零线
        ax_ts.axhline(y=0, color="#AAAAAA", linewidth=0.5, linestyle="--", zorder=0)

        # 年份分隔竖线
        for ys in year_starts:
            if ys <= current_date:
                ax_ts.axvline(x=ys, color="#DDDDDD", linewidth=0.4, linestyle=":",
                              zorder=0)

        # 刻度
        ax_ts.tick_params(labelsize=8)
        ax_ts.yaxis.set_major_locator(MaxNLocator(nbins=5))
        # 统一 X 轴范围与刻度
        ax_ts.set_xlim(date_min, xlim_right)
        if span_days >= 3285:
            ax_ts.xaxis.set_major_locator(mdates.YearLocator())
            ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        else:
            ax_ts.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
            ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        # 隐藏上方两行时间序列的 X 刻度标签
        if i < 2:
            ax_ts.tick_params(axis="x", labelbottom=False)

        # ── 右列：散点图（Ass vs Obs）────────────
        ax_sc = axes[i, 1]

        # 1:1 参考线
        sl = scatter_limits[var]
        ax_sc.plot([sl["xy_min"], sl["xy_max"]], [sl["xy_min"], sl["xy_max"]],
                   color="#AAAAAA", linewidth=0.6, linestyle="--", zorder=0)

        # 动态散点（随着时间累积）
        ax_sc.scatter(
            subset[obs_col], subset[ass_col],
            color=color, s=DOT_SIZE * 0.8, linewidths=0, alpha=0.6, zorder=2,
        )

        # 最新点的十字标记
        if len(subset) > 0:
            last_obs = subset[obs_col].iloc[-1]
            last_ass = subset[ass_col].iloc[-1]
            ax_sc.plot(last_obs, last_ass, marker="+", color=color,
                       markersize=7, markeredgewidth=1.2, zorder=4)

        ax_sc.set_xlim(sl["xy_min"], sl["xy_max"])
        ax_sc.set_ylim(sl["xy_min"], sl["xy_max"])
        ax_sc.set_xlabel(f"{var} Obs ({VAR_UNITS})", fontsize=8, color="black")
        ax_sc.set_ylabel(f"{var} Ass ({VAR_UNITS})", fontsize=8, color="black")
        ax_sc.tick_params(labelsize=7)
        ax_sc.set_box_aspect(1)

    # ── X 轴标签（最下面一行，已在循环内设置locator）───
    ax_ts = axes[-1, 0]
    ax_ts.tick_params(axis="x", labelsize=8, rotation=0)

    fig.subplots_adjust(left=0.09, right=0.97, top=0.97, bottom=0.07,
                        wspace=0.12, hspace=0.25)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  GIF 合成
# ═══════════════════════════════════════════════════════════════

def make_gif(frame_dir: Path, output_path: Path, duration: float = GIF_DURATION,
             scale: float = GIF_SCALE) -> None:
    from PIL import Image

    frames = sorted(frame_dir.glob("*.png"))
    if not frames:
        print("  [WARN] 无帧可合成")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(frames)
    print(f"  加载 {n} 帧并缩放至 {int(scale*100)}% ...")
    images = []
    for fp in tqdm(frames, desc="  加载帧"):
        img = Image.open(fp)
        if scale != 1.0:
            img = img.resize(
                (int(img.width * scale), int(img.height * scale)),
                Image.LANCZOS,
            )
        images.append(img)

    print(f"  合成 GIF ({n} 帧) ...")
    durations = [duration] * (n - 1) + [duration * 10]
    images[0].save(
        output_path, save_all=True, append_images=images[1:],
        duration=durations, loop=0, optimize=True,
    )
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  [OK] GIF: {output_path}  ({size_mb:.1f} MB)")


# ═══════════════════════════════════════════════════════════════
#  Y 轴范围 & 散点范围计算
# ═══════════════════════════════════════════════════════════════

def compute_limits() -> tuple[dict, dict]:
    """从 VAR_RANGES 读取统一范围。"""
    y_limits = {}
    scatter_limits = {}
    for var in ["GPP", "TER", "NEE"]:
        vmin, vmax = VAR_RANGES[var]
        y_limits[var] = (vmin, vmax)
        scatter_limits[var] = {"xy_min": vmin, "xy_max": vmax}
    return y_limits, scatter_limits


# ═══════════════════════════════════════════════════════════════
#  自动计算步长
# ═══════════════════════════════════════════════════════════════

def auto_step(total_days: int, target_sec: float = 10,
              frame_duration: float = 0.12) -> int:
    """根据总天数和目标播放时长，计算最优帧间隔。"""
    n_frames = max(1, int(target_sec / frame_duration))
    step = max(1, total_days // n_frames)
    return step


# ═══════════════════════════════════════════════════════════════
#  Preview（最后一帧）
# ═══════════════════════════════════════════════════════════════

def render_preview(preview_path: Path | None = None,
                  xaxis_mode: str = "fixed") -> dict:
    """生成最后一帧预览，返回渲染参数供后续使用。"""
    _setup_style()

    print("加载通量数据 ...")
    df = load_flux_data(FLUXDATA_PATH)
    print(f"  数据: {len(df)} 天, {df['Date'].min().date()} ~ {df['Date'].max().date()}")

    y_limits, scatter_limits = compute_limits()
    print("  统一范围:")
    for k, v in y_limits.items():
        print(f"    {k}: ({v[0]:.2f}, {v[1]:.2f})")

    if preview_path is None:
        preview_path = PROJECT_ROOT / "frames" / "_preview_flux.png"

    print(f"\n渲染最后一帧预览 → {preview_path}")
    render_frame(df, len(df) - 1, preview_path, y_limits, scatter_limits, xaxis_mode)
    print(f"  [OK] Preview: {preview_path}")

    # 返回参数供后续使用
    return {
        "df": df,
        "y_limits": y_limits,
        "scatter_limits": scatter_limits,
    }


# ═══════════════════════════════════════════════════════════════
#  批量渲染 + GIF
# ═══════════════════════════════════════════════════════════════

def render_all(
    step: int | None = None,
    target_duration: float = TARGET_DURATION,
    make_gif_output: bool = True,
    xaxis_mode: str = "fixed",
) -> None:
    """逐帧渲染 + 合成 GIF。"""
    params = render_preview(xaxis_mode=xaxis_mode)

    df = params["df"]
    y_limits = params["y_limits"]
    scatter_limits = params["scatter_limits"]

    # 自动计算步长
    if step is None:
        step = auto_step(len(df), target_duration, GIF_DURATION)
        print(f"  自动步长: {step} 天/帧 (目标 {target_duration}s, {GIF_DURATION}s/帧)")

    # 清空帧目录
    import shutil
    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    indices = list(range(0, len(df), step))
    if indices[-1] != len(df) - 1:
        indices.append(len(df) - 1)

    print(f"\n渲染 {len(indices)} 帧 (步长={step}天) ...")
    for idx in tqdm(indices, desc="  渲染帧"):
        frame_path = FRAMES_DIR / f"frame_{idx:05d}.png"
        if frame_path.exists():
            continue
        render_frame(df, idx, frame_path, y_limits, scatter_limits, xaxis_mode)

    print(f"\n帧已保存至: {FRAMES_DIR}")

    if make_gif_output:
        print("\n合成 GIF ...")
        gif_path = OUTPUT_DIR / "flux_timeseries.gif"
        make_gif(FRAMES_DIR, gif_path)

    print("\n完成!")


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="通量站点时间序列+散点动图")
    parser.add_argument("--preview", action="store_true", default=False,
                        help="仅生成最后一帧预览")
    parser.add_argument("--step", type=int, default=None,
                        help="帧间隔（天），默认自动根据 --duration 计算")
    parser.add_argument("--duration", type=float, default=TARGET_DURATION,
                        help=f"目标 GIF 播放时长（秒），默认 {TARGET_DURATION}")
    parser.add_argument("--no-gif", action="store_true", default=False,
                        help="不合成 GIF")
    parser.add_argument("--xmode", choices=["fixed", "auto"], default="fixed",
                        help="X轴模式: fixed=全程固定 | auto=随帧扩展")
    args = parser.parse_args()

    # 根据 xmode 调整输出子目录
    global OUTPUT_DIR, FRAMES_DIR
    suffix = "_auto" if args.xmode == "auto" else ""
    OUTPUT_DIR = PROJECT_ROOT / f"output/flux_ts{suffix}"
    FRAMES_DIR = PROJECT_ROOT / f"frames/flux_ts{suffix}"

    if args.preview:
        render_preview(xaxis_mode=args.xmode)
    else:
        render_all(step=args.step, target_duration=args.duration,
                   make_gif_output=not args.no_gif,
                   xaxis_mode=args.xmode)


if __name__ == "__main__":
    main()
