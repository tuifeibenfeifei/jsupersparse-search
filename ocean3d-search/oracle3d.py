# -*- coding: utf-8 -*-
"""oracle3d.py — 3D 溶解氧黑盒查询接口（2D oracle 的三维推广）
黑盒语义：给定 (depth_idx, lat_idx, lon_idx)，返回该格是否低氧目标 + 协变量向量。
与 2D 物种/叶绿素 oracle 同构：presence[K,L,J]、covs[19,K,L,J]，每次查询计 1。
掩膜（陆地下方/无数据）格：presence=False，协变量填全局均值（避免 NaN 污染）。
"""
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data" / "oracle3d.npz"


class Oracle3D:
    """加载预构建 3D 网格 oracle（真实地中海西部溶解氧低氧核）"""

    def __init__(self, npz_path: Path = DATA):
        d = np.load(npz_path)
        self.depth = d["depth"]        # (K,)
        self.lat = d["lat"]            # (L,)
        self.lon = d["lon"]            # (J,)
        self.presence = d["presence"]  # (K, L, J) bool
        self.X = d["X"]                # (19, N)
        self.mask = d["mask"]          # (N,) 有效格
        self.do3d = d["do3d"]          # (K, L, J) umol/l
        self.target_th = float(d["target_th"])
        self.K, self.L, self.J = self.presence.shape
        # 协变量 3D 化（X 已是全网格展平 (19, K*L*J)，含 NaN 列），NaN 填全局均值
        covs3 = self.X.reshape(self.X.shape[0], self.K, self.L, self.J)
        for k in range(covs3.shape[0]):
            v = covs3[k]
            m = np.isfinite(v)
            if m.any():
                covs3[k, ~m] = float(np.nanmean(v[m]))
            else:
                covs3[k] = 0.0
        self.covs = covs3.astype(np.float32)      # (19, K, L, J)
        self.cov_names = [f"cov{i}" for i in range(self.X.shape[0])]
        self.query_count = 0

    def in_bounds(self, k: int, i: int, j: int) -> bool:
        return 0 <= k < self.K and 0 <= i < self.L and 0 <= j < self.J

    def query_cell(self, k: int, i: int, j: int):
        """黑盒查询：越界同样消耗预算，返回 (None, None)"""
        self.query_count += 1
        if not self.in_bounds(k, i, j):
            return None, None
        return bool(self.presence[k, i, j]), self.covs[:, k, i, j].copy()

    @property
    def total_cells(self):
        return self.K * self.L * self.J

    def presence_cells(self):
        return [tuple(c) for c in np.argwhere(self.presence)]

    def reset_counter(self):
        self.query_count = 0
