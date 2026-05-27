"""
Phase 6 v2: 主脚本 — 逐帧渲染 + GIF动画 (MP4已关闭, 含Forest)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    INVENTORY_CSV, COLOR_LIMITS_JSON, OUTPUT_DIR, FRAMES_DIR,
    SCENARIOS, FOREST_TYPES, VARIABLES, YEARS,
    SCENARIO_LABELS, FOREST_LABELS, VARIABLE_LABELS,
    GIF_DURATION, MAKE_MP4,
)
from core import DataLoader, load_and_average, load_forest_average, load_forest_single
from mapper import plot_single_map
from compute_color_limits import compute_global_color_limits, load_color_limits


# ═══════════════════════════════════════════════════════════════
#  动画合成
# ═══════════════════════════════════════════════════════════════

def make_gif(frame_dir: Path, output_path: Path, duration: float = GIF_DURATION) -> None:
    """用 imageio 将帧 PNG 合成为 GIF。"""
    import imageio.v2 as imageio
    from PIL import Image

    frames = sorted(frame_dir.glob("*.png"))
    if not frames:
        print(f"  [WARN] 无帧可合成: {frame_dir}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = []
    for fp in tqdm(frames, desc=f"  合成 GIF: {output_path.name}"):
        images.append(imageio.imread(fp))

    imageio.mimsave(output_path, images, duration=duration, loop=0)
    print(f"  [OK] GIF: {output_path}")


# ═══════════════════════════════════════════════════════════════
#  渲染核心 (v2: Forest支持, 关MP4)
# ═══════════════════════════════════════════════════════════════

def render_animation(
    loader: DataLoader,
    color_limits: dict,
    scenario: str,
    forest_type: str,
    variable: str,
    mode: str,
) -> None:
    limit_key = f"{variable}_{forest_type}"
    clim = color_limits.get(limit_key, {"vmin": 0, "vmax": 10})
    vmin, vmax = clim["vmin"], clim["vmax"]

    lon, lat = loader.lon_lat_grid()

    tag = f"{variable}_{forest_type}_{scenario}"
    if mode == "ensemble":
        tag += "_ensemble"
        model_label = "多模型集合平均"
    else:
        tag += f"_{mode}"
        model_label = mode

    frame_dir = FRAMES_DIR / tag
    frame_dir.mkdir(parents=True, exist_ok=True)

    if mode == "ensemble":
        out_gif = OUTPUT_DIR / "ensemble" / "GIF" / f"{tag}.gif"
    else:
        out_gif = OUTPUT_DIR / "per_model" / mode / "GIF" / f"{tag}.gif"

    print(f"\n{'='*60}")
    print(f"渲染: {VARIABLE_LABELS[variable]} | {FOREST_LABELS.get(forest_type, forest_type)} "
          f"| {SCENARIO_LABELS[scenario]} | {model_label}")
    print(f"{'='*60}")

    for year in tqdm(YEARS, desc="  渲染帧"):
        frame_path = frame_dir / f"frame_{year:04d}.png"
        if frame_path.exists():
            continue

        try:
            if mode == "ensemble":
                if forest_type == "Forest":
                    data = load_forest_average(loader, scenario, variable, year)
                else:
                    data = load_and_average(loader, scenario, forest_type, variable, year)
            else:
                if forest_type == "Forest":
                    data = load_forest_single(loader, mode, scenario, variable, year)
                else:
                    arr = loader.load_single(mode, scenario, forest_type, variable, year)
                    data = arr.filled(np.nan)

            plot_single_map(
                data=data, lon=lon, lat=lat,
                variable=variable, forest_type=forest_type,
                scenario=scenario, year=year,
                model_label=model_label,
                vmin=vmin, vmax=vmax,
                save_path=frame_path,
            )
        except Exception as e:
            print(f"  [ERROR] year={year}: {e}")
            continue

    print(f"\n  合成动画 ...")
    make_gif(frame_dir, out_gif)
    print(f"  帧保留于: {frame_dir}")


# ═══════════════════════════════════════════════════════════════
#  批量模式 (v2: 含Forest)
# ═══════════════════════════════════════════════════════════════

def batch_render(mode: str = "ensemble") -> None:
    loader = DataLoader(INVENTORY_CSV)

    if not COLOR_LIMITS_JSON.exists():
        print("色带范围未计算，正在预计算...")
        compute_global_color_limits(loader)
    color_limits = load_color_limits()

    total = len(VARIABLES) * len(FOREST_TYPES) * len(SCENARIOS)
    if mode != "ensemble":
        total *= len(loader.models())

    print(f"\n批量渲染模式: {mode}")
    print(f"总任务数: {total}")

    for variable in VARIABLES:
        for forest in FOREST_TYPES:
            for scenario in SCENARIOS:
                if mode == "ensemble":
                    render_animation(loader, color_limits, scenario, forest, variable, mode="ensemble")
                else:
                    for model in loader.models():
                        render_animation(loader, color_limits, scenario, forest, variable, mode=model)

    print("\n" + "=" * 60)
    print("全部渲染完成!")
    print(f"输出目录: {OUTPUT_DIR}")


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CMIP6 动态地图渲染 v2")
    parser.add_argument("--mode", choices=["ensemble", "single"], default="ensemble")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--variable", choices=VARIABLES)
    parser.add_argument("--forest", choices=FOREST_TYPES)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--model", type=str)
    parser.add_argument("--limits-only", action="store_true")
    args = parser.parse_args()

    loader = DataLoader(INVENTORY_CSV)

    if args.limits_only:
        compute_global_color_limits(loader)
        return

    if args.batch:
        batch_render(mode="ensemble" if args.mode == "ensemble" else "single")
        return

    if not all([args.variable, args.forest, args.scenario]):
        parser.error("非批量模式需指定 --variable --forest --scenario")

    if not COLOR_LIMITS_JSON.exists():
        compute_global_color_limits(loader)
    color_limits = load_color_limits()

    mode = "ensemble" if args.mode == "ensemble" else args.model
    if not mode:
        parser.error("单模型模式需指定 --model")

    render_animation(loader, color_limits, args.scenario, args.forest, args.variable, mode=mode)

    if not COLOR_LIMITS_JSON.exists():
        compute_global_color_limits(loader)
    color_limits = load_color_limits()

    mode = "ensemble" if args.mode == "ensemble" else args.model
    if not mode:
        parser.error("单模型模式需指定 --model")

    render_animation(
        loader, color_limits,
        args.scenario, args.forest, args.variable,
        mode=mode,
    )


if __name__ == "__main__":
    main()
