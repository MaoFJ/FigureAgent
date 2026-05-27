"""
Phase 5: 全局色带范围预计算
遍历所有数据，对每个 (variable, forest_type) 计算统一 vmin/vmax
确保跨模型、跨情景、跨年份的颜色可比。
"""

import json
import numpy as np
from tqdm import tqdm
from core import DataLoader
from config import COLOR_LIMITS_JSON, INVENTORY_CSV, DIVERGING_VARS


def compute_global_color_limits(loader: DataLoader) -> dict:
    """
    v2: 含 Forest (阔叶+针叶合计), NEE 强制 vmin=0
    """
    from core import load_forest_average

    limits = {}

    # 原生森林类型
    combos = []
    for variable in loader.variables():
        for forest in loader.forest_types():
            combos.append((variable, forest, False))
    # 新增 Forest = Broadleaf + Needleleaf
    for variable in loader.variables():
        combos.append((variable, "Forest", True))

    for variable, forest, is_forest in combos:
        key = f"{variable}_{forest}"
        print(f"\n计算色带范围: {key} ...")

        all_values = []
        scenarios = loader.scenarios()
        years = loader.years()

        for scenario in tqdm(scenarios, desc=f"  {key}"):
            for year in years:
                try:
                    if is_forest:
                        mean_2d = load_forest_average(loader, scenario, variable, year)
                    else:
                        stack = loader.load_stack(scenario, forest, variable, year)
                        mean_2d = np.nanmean(stack, axis=0)
                    valid = mean_2d[~np.isnan(mean_2d)]
                    if len(valid) > 0:
                        all_values.append(valid)
                except Exception:
                    continue

        if not all_values:
            print(f"  [WARN] {key}: 无有效数据")
            limits[key] = {"vmin": 0, "vmax": 1}
            continue

        all_values = np.concatenate(all_values)
        p_low = np.percentile(all_values, 2)
        p_high = np.percentile(all_values, 98)

        if variable == "NEE":
            vmin = 0.0           # NEE 强制 vmin=0
            vmax = p_high
        else:
            vmin = max(p_low, 0)
            vmax = p_high

        vmin, vmax = _round_nice(vmin, vmax)
        limits[key] = {"vmin": float(vmin), "vmax": float(vmax)}
        print(f"    vmin={vmin:.2f}  vmax={vmax:.2f}  "
              f"(P2={p_low:.2f} P98={p_high:.2f})")

    with open(COLOR_LIMITS_JSON, "w", encoding="utf-8") as f:
        json.dump(limits, f, indent=2, ensure_ascii=False)
    print(f"\n色带范围已保存: {COLOR_LIMITS_JSON}")
    return limits


def _round_nice(vmin: float, vmax: float) -> tuple[float, float]:
    """将范围取整到美观的数值。"""
    span = vmax - vmin
    if span == 0:
        return vmin, vmax + 1

    magnitude = 10 ** np.floor(np.log10(span))
    # 归一化到 [1, 10)
    norm_span = span / magnitude

    if norm_span <= 1.5:
        step = 0.2 * magnitude
    elif norm_span <= 3:
        step = 0.5 * magnitude
    elif norm_span <= 7:
        step = magnitude
    else:
        step = 2 * magnitude

    vmin_r = np.floor(vmin / step) * step
    vmax_r = np.ceil(vmax / step) * step
    return vmin_r, vmax_r


def load_color_limits() -> dict:
    """从 JSON 文件加载色带范围。"""
    with open(COLOR_LIMITS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    loader = DataLoader(INVENTORY_CSV)
    compute_global_color_limits(loader)
