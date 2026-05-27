"""
Phase 3: 核心数据加载引擎 — DataLoader + 集合平均
"""

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds

from config import INVENTORY_CSV


class DataLoader:
    """按 (scenario, forest_type, variable, year) 加载全部模型栅格数据。"""

    def __init__(self, inventory_path: Path | None = None):
        if inventory_path is None:
            inventory_path = INVENTORY_CSV
        self.df = pd.read_csv(inventory_path)
        self._cache: dict[tuple, np.ndarray] = {}

        # 延迟初始化地理元数据（首个 tif 读取后填充）
        self._geo_meta: dict | None = None

    # ── 查询接口 ──────────────────────────────────────

    def models(self) -> list[str]:
        return sorted(self.df["model"].unique())

    def scenarios(self) -> list[str]:
        return sorted(self.df["scenario"].unique())

    def forest_types(self) -> list[str]:
        return sorted(self.df["forest_type"].unique())

    def variables(self) -> list[str]:
        return sorted(self.df["variable"].unique())

    def years(self) -> list[int]:
        return sorted(self.df["year"].unique())

    # ── 核心加载 ──────────────────────────────────────

    def load_stack(
        self,
        scenario: str,
        forest_type: str,
        variable: str,
        year: int,
    ) -> np.ndarray:
        """
        加载所有模型的该维度数据，返回 (N_models, lat, lon) 的 masked array。

        缺失模型 → NaN 填充。
        """
        subset = self.df[
            (self.df["scenario"] == scenario)
            & (self.df["forest_type"] == forest_type)
            & (self.df["variable"] == variable)
            & (self.df["year"] == year)
        ]
        if subset.empty:
            raise ValueError(
                f"无数据: scenario={scenario} forest={forest_type} "
                f"var={variable} year={year}"
            )

        arrays = []
        for _, row in subset.iterrows():
            arr = self._read_tif(row["filepath"])
            arrays.append(arr)

        # 确保所有数组形状一致 → 用首个数组形状对齐
        ref_shape = arrays[0].shape
        aligned = []
        for arr in arrays:
            if arr.shape == ref_shape:
                aligned.append(arr.filled(np.nan))
            else:
                aligned.append(np.full(ref_shape, np.nan, dtype=np.float32))

        return np.array(aligned, dtype=np.float32)

    def load_single(
        self,
        model: str,
        scenario: str,
        forest_type: str,
        variable: str,
        year: int,
    ) -> np.ma.MaskedArray:
        """加载单个模型的栅格数据（masked array）。"""
        subset = self.df[
            (self.df["model"] == model)
            & (self.df["scenario"] == scenario)
            & (self.df["forest_type"] == forest_type)
            & (self.df["variable"] == variable)
            & (self.df["year"] == year)
        ]
        if subset.empty:
            raise ValueError(f"无数据: model={model} ... year={year}")
        return self._read_tif(subset.iloc[0]["filepath"])

    # ── 地理元数据 ────────────────────────────────────

    @property
    def geo_meta(self) -> dict:
        """返回 {crs, transform, width, height, bounds}。"""
        if self._geo_meta is None:
            self._init_geo_meta()
        return self._geo_meta

    def lon_lat_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 (lon_1d, lat_1d) 一维 cell 坐标数组。"""
        meta = self.geo_meta
        transform = meta["transform"]
        width = meta["width"]
        height = meta["height"]

        # 从 transform 直接计算坐标
        lon_1d = np.array([
            transform.c + col * transform.a
            for col in range(width + 1)  # +1 for pcolormesh edges
        ], dtype=np.float64)
        lat_1d = np.array([
            transform.f + row * transform.e
            for row in range(height + 1)
        ], dtype=np.float64)
        return lon_1d, lat_1d

    # ── 内部 ──────────────────────────────────────────

    def _read_tif(self, filepath: str) -> np.ma.MaskedArray:
        cache_key = filepath
        if cache_key in self._cache:
            return self._cache[cache_key]

        with rasterio.open(filepath) as src:
            data = src.read(1, masked=True)
            self._cache[cache_key] = data
        return data

    def _init_geo_meta(self) -> None:
        """从首个 tif 提取地理元数据。"""
        row = self.df.iloc[0]
        with rasterio.open(row["filepath"]) as src:
            self._geo_meta = {
                "crs": str(src.crs),
                "transform": src.transform,
                "width": src.width,
                "height": src.height,
                "bounds": src.bounds,
            }


# ── 集合平均 ──────────────────────────────────────────

def compute_ensemble_mean(stack: np.ndarray) -> np.ndarray:
    """
    stack: (N_models, lat, lon)
    返回 (lat, lon)，忽略 NaN 的均值。
    """
    with np.errstate(all="ignore"):
        mean = np.nanmean(stack, axis=0)
    return mean


def compute_ensemble_std(stack: np.ndarray) -> np.ndarray:
    """集合标准差（模型间离散度）。"""
    with np.errstate(all="ignore"):
        std = np.nanstd(stack, axis=0)
    return std


# ── 便捷函数 ──────────────────────────────────────────

def load_and_average(
    loader: DataLoader,
    scenario: str,
    forest_type: str,
    variable: str,
    year: int,
) -> np.ndarray:
    """一步加载 + 集合平均 → 2D (lat, lon)。"""
    stack = loader.load_stack(scenario, forest_type, variable, year)
    return compute_ensemble_mean(stack)


# ── Forest = Broadleaf + Needleleaf (v2) ──────────────

def load_forest_average(
    loader: DataLoader,
    scenario: str,
    variable: str,
    year: int,
) -> np.ndarray:
    """加载阔叶+针叶的集合平均之和（Forest = Broadleaf + Needleleaf）。"""
    stack_bl = loader.load_stack(scenario, "Broadleaf", variable, year)
    stack_nl = loader.load_stack(scenario, "Needleleaf", variable, year)
    mean_bl = compute_ensemble_mean(stack_bl)
    mean_nl = compute_ensemble_mean(stack_nl)
    return mean_bl + mean_nl


def load_forest_single(
    loader: DataLoader,
    model: str,
    scenario: str,
    variable: str,
    year: int,
) -> np.ndarray:
    """单模型的 Forest = 该模型阔叶 + 针叶。"""
    arr_bl = loader.load_single(model, scenario, "Broadleaf", variable, year)
    arr_nl = loader.load_single(model, scenario, "Needleleaf", variable, year)
    return arr_bl.filled(np.nan) + arr_nl.filled(np.nan)
