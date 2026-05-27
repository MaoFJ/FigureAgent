"""审计C: 快速渲染2帧测试新功能"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import DataLoader, load_and_average, load_forest_average
from mapper import plot_single_map
from compute_color_limits import load_color_limits
from config import INVENTORY_CSV

loader = DataLoader(INVENTORY_CSV)
clim = load_color_limits()
lon, lat = loader.lon_lat_grid()

test_dir = Path(r"d:\Programs\FigureAgent\frames\_test_v2")
test_dir.mkdir(parents=True, exist_ok=True)

# Test 1: NEE Forest ssp585 2060 (验证Forest + NEE vmin=0 + 新色带)
key = "NEE_Forest"
cl = clim[key]
print(f"{key}: vmin={cl['vmin']} vmax={cl['vmax']}")
data = load_forest_average(loader, "ssp585", "NEE", 2060)
print(f"  data range: [{np.nanmin(data):.1f}, {np.nanmax(data):.1f}]")
plot_single_map(data, lon, lat, "NEE", "Forest", "ssp585", 2060, "test",
                cl["vmin"], cl["vmax"],
                save_path=test_dir / "NEE_Forest_ssp585_2060.png")
print("Test 1 OK")

# Test 2: GPP Broadleaf ssp126 2015 (验证基本功能 + 新布局)
key = "GPP_Broadleaf"
cl = clim[key]
data = load_and_average(loader, "ssp126", "Broadleaf", "GPP", 2015)
plot_single_map(data, lon, lat, "GPP", "Broadleaf", "ssp126", 2015, "test",
                cl["vmin"], cl["vmax"],
                save_path=test_dir / "GPP_Broadleaf_ssp126_2015.png")
print("Test 2 OK")

print("All audit tests passed!")
