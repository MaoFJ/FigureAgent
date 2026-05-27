"""
快速测试：渲染 3 帧验证管线是否正常
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from core import DataLoader, load_and_average
from mapper import plot_single_map
from compute_color_limits import load_color_limits
from config import INVENTORY_CSV, YEARS

# 测试组合：NEE + Broadleaf + ssp585（变化最显著）
TEST_VAR = "NEE"
TEST_FOREST = "Broadleaf"
TEST_SCENARIO = "ssp585"
TEST_YEARS = [2015, 2037, 2060]

loader = DataLoader(INVENTORY_CSV)
color_limits = load_color_limits()

limit_key = f"{TEST_VAR}_{TEST_FOREST}"
clim = color_limits[limit_key]
vmin, vmax = clim["vmin"], clim["vmax"]
print(f"色带范围: {vmin} ~ {vmax}")

lon, lat = loader.lon_lat_grid()
print(f"网格: lon=[{lon.min():.2f}, {lon.max():.2f}] ({len(lon)}), "
      f"lat=[{lat.min():.2f}, {lat.max():.2f}] ({len(lat)})")

test_dir = Path(r"d:\Programs\FigureAgent\frames\_test")
test_dir.mkdir(parents=True, exist_ok=True)

for year in TEST_YEARS:
    print(f"\n渲染 {year} ...")
    data = load_and_average(loader, TEST_SCENARIO, TEST_FOREST, TEST_VAR, year)
    print(f"  数据范围: [{np.nanmin(data):.2f}, {np.nanmax(data):.2f}] "
          f"均值: {np.nanmean(data):.2f}")

    fig = plot_single_map(
        data=data, lon=lon, lat=lat,
        variable=TEST_VAR, forest_type=TEST_FOREST,
        scenario=TEST_SCENARIO, year=year,
        model_label="多模型集合平均 (6 models)",
        vmin=vmin, vmax=vmax,
        save_path=test_dir / f"test_{year}.png",
    )

print(f"\n测试帧已保存: {test_dir}")
