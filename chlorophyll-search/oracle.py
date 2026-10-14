# -*- coding: utf-8 -*-
"""oracle.py — 稀有物种黑盒查询接口（物种版 cubiomes_wrapper + database）
黑盒语义：给定 (lat, lon)，返回该位置是否出现目标物种 + 该位置的协变量向量。
每次查询计入 query_count。与用户框架 cubiomes_wrapper.get_temperature/is_deep_ocean 同构：
    cubiomes:  (x,z) → (温度, 是否深海)
    species:   (lat,lon) → (协变量, 是否出现)
"""
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data" / "oracle.npz"


class SpeciesOracle:
    """加载预构建的网格 oracle，提供黑盒查询"""

    def __init__(self, npz_path: Path = DATA):
        d = np.load(npz_path)
        self.lon0 = float(d["lon0"])
        self.lat0 = float(d["lat0"])
        self.cell = float(d["cell_deg"])
        self.ncols = int(d["ncols"])
        self.nrows = int(d["nrows"])
        self.presence = d["presence"]          # (nrows, ncols) bool
        self.covs = d["covs"]                  # (K, nrows, ncols) float32
        self.cov_names = [str(x) for x in d["cov_names"]]
        self.study_poly = d["study_poly"]      # [lon_min, lon_max, lat_min, lat_max]
        self.query_count = 0

    # ---- 坐标换算 ----
    def latlon_to_cell(self, lat: float, lon: float):
        i = int((lat - self.lat0) / self.cell)
        j = int((lon - self.lon0) / self.cell)
        if i < 0 or j < 0 or i >= self.nrows or j >= self.ncols:
            return None
        return i, j

    def cell_to_latlon(self, i: int, j: int):
        return self.lat0 + (i + 0.5) * self.cell, self.lon0 + (j + 0.5) * self.cell

    def in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self.nrows and 0 <= j < self.ncols

    # ---- 黑盒查询：一次查询 = 一次计数 ----
    def query(self, lat: float, lon: float):
        """返回 (presence: bool, cov_vector: np.ndarray)；越界返回 (None, None)"""
        cell = self.latlon_to_cell(lat, lon)
        if cell is None:
            return None, None
        i, j = cell
        self.query_count += 1
        return bool(self.presence[i, j]), self.covs[:, i, j].copy()

    def query_cell(self, i: int, j: int):
        """网格查询：越界查询同样消耗一次预算（黑盒语义），返回 (None, None)"""
        self.query_count += 1
        if not self.in_bounds(i, j):
            return None, None
        return bool(self.presence[i, j]), self.covs[:, i, j].copy()

    # ---- 供实验使用的统计 ----
    @property
    def total_cells(self):
        return self.nrows * self.ncols

    def presence_cells(self):
        return [tuple(c) for c in np.argwhere(self.presence)]

    def reset_counter(self):
        self.query_count = 0


def covariate_guidance_table(oracle: SpeciesOracle, found_cells, n_bins: int = 6):
    """Stage2 数据积累：统计已发现出现网格的协变量分位数区间（参数→出现概率导向表）
    返回 {cov_index: (lo, hi)}：出现网格的中部区间（25%~75% 分位），供导向打分
    """
    if not found_cells:
        return None
    pts = np.array([[oracle.covs[k, i, j] for i, j in found_cells]
                    for k in range(oracle.covs.shape[0])])  # (K, n)
    table = {}
    for k in range(oracle.covs.shape[0]):
        v = pts[k]
        lo = np.percentile(v, 25)
        hi = np.percentile(v, 75)
        if hi - lo < 1e-9:
            lo, hi = np.min(v), np.max(v)
        table[k] = (float(lo), float(hi))
    return table


def guidance_score(oracle: SpeciesOracle, i: int, j: int, table, cov_indices):
    """导向打分：协变量落在出现区间内的比例（0~1）"""
    if table is None:
        return 0.0
    hit = 0
    for k in cov_indices:
        lo, hi = table[k]
        v = float(oracle.covs[k, i, j])
        if lo <= v <= hi:
            hit += 1
    return hit / len(cov_indices)
