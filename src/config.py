"""
CMIP6 NEP/GPP/TER 多情景动态地图可视化 — 全局配置 (v2)
"""

from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

# === 数据路径 ===
DATA_ROOTS = [
    Path(r"E:\Personal\2023_Tengjiayin\CorrectCIMP6\100"),
    Path(r"E:\Personal\2023_Tengjiayin\CorrectCIMP6\250correct"),
]

# 跳过的模型（空目录）
SKIP_MODELS = {"IPSL-CM6A-LR"}

# === 情景映射 ===
SCENARIO_LABELS = {
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}
SCENARIOS = list(SCENARIO_LABELS.keys())
SCENARIO_SHORT = {"ssp126": "SSP126", "ssp245": "SSP245", "ssp585": "SSP585"}

# === 森林类型 (Forest = Broadleaf + Needleleaf) ===
FOREST_TYPES = ["Broadleaf", "Needleleaf", "Forest"]
FOREST_LABELS = {"Broadleaf": "阔叶林", "Needleleaf": "针叶林", "Forest": "森林"}

# === 变量 ===
VARIABLES = ["GPP", "TER", "NEE"]
VARIABLE_LABELS = {
    "GPP": "总初级生产力 GPP",
    "TER": "生态系统总呼吸 TER",
    "NEE": "净生态系统生产力 NEP",
}
VARIABLE_UNITS = {"GPP": "GPP (gC/m2/yr)", "TER": "TER (gC/m2/yr)", "NEE": "NEE (gC/m2/yr)"}

# === 色带配置 ===
CMAPS = {
    "GPP": "YlGn",
    "TER": "Oranges",
    "NEE": LinearSegmentedColormap.from_list(
        "NEP_cmap", ["#FFFACD", "#9ACD32", "#006400"]
    ),  # 柠檬黄→黄绿→深绿 (与GPP区分)
}
DIVERGING_VARS = set()  # NEE 不再 diverging，强制 vmin=0

# === 年份范围 ===
YEAR_START = 2015
YEAR_END = 2060
YEARS = list(range(YEAR_START, YEAR_END + 1))

# === 输出路径 ===
OUTPUT_DIR = Path(r"d:\Programs\FigureAgent\output")
FRAMES_DIR = Path(r"d:\Programs\FigureAgent\frames")
INVENTORY_CSV = Path(r"d:\Programs\FigureAgent\data_inventory.csv")
COLOR_LIMITS_JSON = Path(r"d:\Programs\FigureAgent\color_limits.json")

# === 动画参数 ===
FPS = 3
GIF_DURATION = 0.455   # 0.35 * 1.3 = 延长30%
MAKE_MP4 = False        # 用户自行输出MP4

# === 地图参数 ===
FIGURE_DPI = 150
FIGURE_WIDTH = 12
FIGURE_HEIGHT = 7
BG_COLOR = "white"
MAP_PAD_RATIO = 0.06          # 边界扩展比例

# === 中文字体 ===
# 按优先级尝试
CJK_FONT_CANDIDATES = ["Noto Sans SC", "Microsoft YaHei", "SimHei", "STKaiti"]
# 若均不可用则回退英文标签

# === 并行/批处理 ===
MAX_WORKERS = 4  # 并行渲染帧数
