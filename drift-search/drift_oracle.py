# -*- coding: utf-8 -*-
"""drift_oracle.py — 海上搜救逆向漂移 黑盒 oracle（接口对齐 SpeciesOracle）。

  - .nrows / .ncols / .covs (K,nrows,ncols) / .cov_names
  - .in_bounds(i,j)
  - .query_cell(i,j) -> (presence: bool|None, cov_vector: ndarray|None)
  - .query_count / .reset_counter() / .total_cells / .presence_cells

阶段 A：presence = 冻结残骸快照（距任一残骸 ≤ R_A 格）。
阶段 B：presence = 失事点 P0 静态证据（距 P0 ≤ R_B 格）。
阶段 B 运行时通过 append_covariate() 追加第 20 个协变量（到解码质心距离）。
真值（P0/残骸位置/物理参数）仅报告与解码 wrapper 读取，不喂搜索器。
"""
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"


class DriftOracle:
    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        self.nrows = int(d["nrows"])
        self.ncols = int(d["ncols"])
        self.presence = d["presence"].astype(bool)       # (nrows,ncols)
        self.covs = d["covs"].astype(np.float32)         # (K,nrows,ncols)
        self.cov_names = [str(x) for x in d["cov_names"]]
        self.cell_km = float(d["cell_km"])
        self.query_count = 0
        # 真值（仅报告/解码用）
        self.P0_i = int(d["P0_i"]); self.P0_j = int(d["P0_j"])
        self.P0_x = float(d["P0_x"]); self.P0_y = float(d["P0_y"])
        self.meta = str(d["meta"])
        # stageA 专属（stageB 无）
        for k in ("debris_cell", "debris_km", "R_A_cell", "R_B_cell",
                  "T_A_h", "cu_p0", "cv_p0", "wu_mean", "wv_mean",
                  "leeway_hat", "cloud_center_x", "cloud_center_y"):
            if k in d.files:
                v = d[k]
                setattr(self, k, float(v) if v.ndim == 0 else v)

    def append_covariate(self, band: np.ndarray, name: str):
        """运行时追加一个协变量波段（阶段 B 的'到解码质心距离'）。"""
        band = band.astype(np.float32).reshape(1, self.nrows, self.ncols)
        self.covs = np.concatenate([self.covs, band], axis=0)
        self.cov_names.append(name)

    @property
    def total_cells(self):
        return self.nrows * self.ncols

    def in_bounds(self, i, j):
        return 0 <= i < self.nrows and 0 <= j < self.ncols

    def query_cell(self, i, j):
        """网格查询：越界同样消耗一次预算（黑盒语义），返回 (None,None)。"""
        self.query_count += 1
        if not self.in_bounds(i, j):
            return None, None
        return bool(self.presence[i, j]), self.covs[:, i, j].copy()

    def presence_cells(self):
        return [tuple(c) for c in np.argwhere(self.presence)]

    def reset_counter(self):
        self.query_count = 0
