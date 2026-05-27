"""
Phase 2: 扫描数据目录，构建完整数据清单 CSV
"""

import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from config import (
    DATA_ROOTS, SKIP_MODELS, SCENARIOS, FOREST_TYPES,
    VARIABLES, YEARS, INVENTORY_CSV,
)


def scan_data() -> pd.DataFrame:
    """扫描所有数据根目录，返回包含所有 tif 文件的 DataFrame。"""
    records = []

    for root in DATA_ROOTS:
        if not root.exists():
            print(f"[WARN] 数据目录不存在: {root}")
            continue

        resolution = "100km" if root.name == "100" else "250km"

        for model_dir in sorted(root.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            if model in SKIP_MODELS:
                print(f"[SKIP] 跳过模型: {model}")
                continue

            for scenario in SCENARIOS:
                scenario_dir = model_dir / scenario
                if not scenario_dir.exists():
                    continue

                for forest in FOREST_TYPES:
                    forest_dir = scenario_dir / forest
                    if not forest_dir.exists():
                        continue

                    # 进入 0_251/Region/Ann/ 目录
                    # 兼容不同内部路径结构
                    tif_dir = _find_ann_dir(forest_dir)
                    if tif_dir is None:
                        continue

                    for var in VARIABLES:
                        for year in YEARS:
                            fname = f"{var}_{year}.tif"
                            fpath = tif_dir / fname
                            if fpath.exists():
                                records.append({
                                    "model": model,
                                    "resolution": resolution,
                                    "scenario": scenario,
                                    "forest_type": forest,
                                    "variable": var,
                                    "year": year,
                                    "filepath": str(fpath),
                                })

    df = pd.DataFrame(records)
    print(f"\n扫描完成: 共找到 {len(df)} 个文件")
    print(f"  模型: {df['model'].nunique()} 个")
    print(f"  情景: {df['scenario'].unique().tolist()}")
    print(f"  森林类型: {df['forest_type'].unique().tolist()}")
    print(f"  变量: {df['variable'].unique().tolist()}")
    print(f"  年份: {df['year'].min()} - {df['year'].max()}")

    # 按模型分组统计
    print("\n各模型文件数:")
    for model, grp in df.groupby("model"):
        print(f"  {model}: {len(grp)}")

    # 保存
    df.to_csv(INVENTORY_CSV, index=False, encoding="utf-8-sig")
    print(f"\n数据清单已保存: {INVENTORY_CSV}")
    return df


def _find_ann_dir(forest_dir: Path) -> Path | None:
    """自动查找 {forest_dir} 下的 Ann/ 目录。"""
    # 直接尝试常见路径
    candidates = [
        forest_dir / "0_251" / "Region" / "Ann",
        forest_dir / "0_100" / "Region" / "Ann",
    ]
    for c in candidates:
        if c.exists():
            return c
    # 递归搜索 Ann/ 目录（最多3层）
    for p in forest_dir.rglob("Ann"):
        if p.is_dir():
            return p
    return None


def validate_tif_samples(df: pd.DataFrame, n: int = 3) -> None:
    """随机抽样验证 tif 的 CRS、分辨率、数值范围。"""
    import rasterio
    samples = df.sample(min(n, len(df)))

    print("\n=== 抽样验证 GeoTIFF ===")
    for _, row in samples.iterrows():
        with rasterio.open(row["filepath"]) as src:
            print(f"\n文件: {Path(row['filepath']).name}")
            print(f"  模型={row['model']} 情景={row['scenario']} "
                  f"森林={row['forest_type']} 变量={row['variable']} 年={row['year']}")
            print(f"  CRS: {src.crs}")
            print(f"  分辨率: {src.res}")
            print(f"  形状: {src.shape}")
            print(f"  NoData: {src.nodata}")
            data = src.read(1, masked=True)
            print(f"  有效值范围: [{data.min():.4f}, {data.max():.4f}]")
            print(f"  均值: {data.mean():.4f}")


if __name__ == "__main__":
    df = scan_data()
    validate_tif_samples(df)
