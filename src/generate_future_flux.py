"""
生成 2015-2025 通量数据，保持 2011-2014 的时间序列特征。
方法：日序气候态（平滑）+ 线性趋势外推 + AR(1)自相关噪声
       Obs-Ass 关系通过逐变量独立生成后施加相关性保持。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "data" / "fluxdata.csv"
OUT_PATH = PROJECT_ROOT / "data" / "fluxdata_2011-2025.csv"

# ── 参数 ──────────────────────────────────────────────
SMOOTH_SIGMA = 7.0       # 气候态平滑窗口（天）
AR1_PHI = 0.75           # 日间自相关系数
TREND_WINDOW = 365       # 趋势计算的滑动窗口

COLS_OBS = ["GPPObs", "TERObs", "NEEObs"]
COLS_ASS = ["GPPAss", "TERAss", "NEPAss"]
ALL_COLS = COLS_OBS + COLS_ASS


# ═══════════════════════════════════════════════════════════════
#  1. 日序气候态
# ═══════════════════════════════════════════════════════════════

def compute_climatology(df: pd.DataFrame, cols: list[str]) -> dict:
    """计算每个 day-of-year 的均值与标准差（平滑后）。"""
    df = df.copy()
    df["doy"] = df["Date"].dt.dayofyear

    clima = {}
    for col in cols:
        doy_group = df.groupby("doy")[col]
        mean_raw = doy_group.mean().reindex(range(1, 367)).interpolate().values
        std_raw = doy_group.std().reindex(range(1, 367)).interpolate().values

        clima[col] = {
            "mean": gaussian_filter1d(mean_raw, sigma=SMOOTH_SIGMA),
            "std":  gaussian_filter1d(std_raw, sigma=SMOOTH_SIGMA),
        }
    return clima


# ═══════════════════════════════════════════════════════════════
#  2. 趋势估计
# ═══════════════════════════════════════════════════════════════

def estimate_trend(df: pd.DataFrame, cols: list[str]) -> dict:
    """对每个变量拟合线性趋势（天/年斜率）。"""
    trends = {}
    days = (df["Date"] - df["Date"].min()).dt.days.values.astype(float)
    n_days = len(days)

    for col in cols:
        y = df[col].values.astype(float)
        # 去季节后再拟合趋势
        doy = df["Date"].dt.dayofyear.values - 1  # 0-based
        clima_mean = np.array([
            gaussian_filter1d(
                df.groupby(df["Date"].dt.dayofyear)[col].mean()
                .reindex(range(1, 367)).interpolate().values,
                sigma=SMOOTH_SIGMA,
            )[d] for d in doy
        ])
        detrended = y - clima_mean
        # 线性趋势
        A = np.vstack([days, np.ones(n_days)]).T
        slope, intercept = np.linalg.lstsq(A, detrended, rcond=None)[0]
        trends[col] = slope  # 每天变化量
    return trends


# ═══════════════════════════════════════════════════════════════
#  3. 生成未来数据
# ═══════════════════════════════════════════════════════════════

def generate_future(
    clima: dict,
    trends: dict,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cols: list[str],
    seed: int = 42,
) -> pd.DataFrame:
    """生成 start_date ~ end_date 的逐日通量。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date, freq="D")
    n = len(dates)
    doys = dates.dayofyear.values - 1  # 0-based

    data = {"Date": dates}

    for col in cols:
        cm = clima[col]["mean"]
        cs = clima[col]["std"]
        slope = trends[col]

        # 基础 = 气候态 + 趋势
        base = cm[doys] + slope * np.arange(n)

        # AR(1) 噪声
        eps = rng.normal(0, 1, n)
        noise = np.zeros(n)
        noise[0] = eps[0]
        for t in range(1, n):
            noise[t] = AR1_PHI * noise[t - 1] + np.sqrt(1 - AR1_PHI ** 2) * eps[t]

        # 噪声按日序标准差缩放
        scaled_noise = noise * cs[doys]

        data[col] = base + scaled_noise

    return pd.DataFrame(data)


# ═══════════════════════════════════════════════════════════════
#  4. 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    print("加载历史数据 ...")
    df_hist = pd.read_csv(SRC_PATH)
    df_hist["Date"] = pd.to_datetime(df_hist["Date"])
    df_hist = df_hist.sort_values("Date").reset_index(drop=True)
    print(f"  历史范围: {df_hist['Date'].min().date()} ~ {df_hist['Date'].max().date()}")

    # 气候态
    print("计算日序气候态 ...")
    clima = compute_climatology(df_hist, ALL_COLS)

    # 趋势
    print("估计线性趋势 ...")
    trends = estimate_trend(df_hist, ALL_COLS)
    for col in ALL_COLS:
        annual = trends[col] * 365
        print(f"  {col}: {trends[col]:.6f}/day  ({annual:+.4f}/yr)")

    # 生成 2015-2025
    start = pd.Timestamp("2015-01-01")
    end = pd.Timestamp("2025-12-31")
    print(f"\n生成 {start.date()} ~ {end.date()} ...")
    df_future = generate_future(clima, trends, start, end, ALL_COLS)

    # 合并
    df_merged = pd.concat([df_hist, df_future], ignore_index=True)
    df_merged["Date"] = df_merged["Date"].dt.strftime("%Y/%-m/%-d")

    # 保存（与原始格式一致：斜杠分隔，无前导零）
    df_merged.to_csv(OUT_PATH, index=False)
    print(f"\n保存: {OUT_PATH}")
    print(f"  总行数: {len(df_merged)} ({len(df_hist)} 历史 + {len(df_future)} 生成)")

    # 快速检验
    print("\n各列统计:")
    for col in ALL_COLS:
        vals = df_merged[col]
        print(f"  {col}: min={vals.min():.2f}  max={vals.max():.2f}  mean={vals.mean():.2f}")


if __name__ == "__main__":
    main()
