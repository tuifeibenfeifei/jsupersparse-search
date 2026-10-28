# -*- coding: utf-8 -*-
"""oracle3d.py — 3D dissolved-oxygen black-box query interface (3D extension of the 2D oracle)
Black-box semantics: given (depth_idx, lat_idx, lon_idx), return whether the cell is a hypoxia target + the covariate vector.
Isomorphic to the 2D species/chlorophyll oracle: presence[K,L,J], covs[19,K,L,J]; each query counts as 1.
Masked cells (below land / no data): presence=False, covariates filled with the global mean (to avoid NaN contamination).
"""
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data" / "oracle3d.npz"


class Oracle3D:
    """load the pre-built 3D grid oracle (real western Mediterranean dissolved-oxygen hypoxia core)"""

    def __init__(self, npz_path: Path = DATA):
        d = np.load(npz_path)
        self.depth = d["depth"]        # (K,)
        self.lat = d["lat"]            # (L,)
        self.lon = d["lon"]            # (J,)
        self.presence = d["presence"]  # (K, L, J) bool
        self.X = d["X"]                # (19, N)
        self.mask = d["mask"]          # (N,) valid cells
        self.do3d = d["do3d"]          # (K, L, J) umol/l
        self.target_th = float(d["target_th"])
        self.K, self.L, self.J = self.presence.shape
        # reshape covariates to 3D (X is already the flattened full grid (19, K*L*J), with NaN columns); fill NaN with the global mean
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
        """black-box query: out-of-bounds also consumes budget; returns (None, None)"""
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
