# -*- coding: utf-8 -*-
# --- repository root (auto-located) ---
import os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve()
for _ in range(8):
    if (REPO_ROOT / 'configs').is_dir():
        break
    REPO_ROOT = REPO_ROOT.parent
# --------------------------------------
"""motion_extrap_smoke.py — proves the improvement-3 module actually works under "drifting targets whose hits can accumulate".

Synthetic dynamic oracle: 100x100, a single target starting at (50,8) drifting at (0, +0.12 columns/query),
detection radius R=3. Under the static assumption the framework circles around old hit points and is left behind by the target;
with motion extrapolation enabled it should chase ahead of the target along the drift direction. seeds 101-110 compare motion off/on.

Usage: python -X utf8 motion_extrap_smoke.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(REPO_ROOT)
sys.path.insert(0, str(ROOT / "species-search"))
from species_agent import SpeciesSearcher  # noqa: E402

N = 100
R = 3.0
VJ = 0.12          # column-direction drift velocity (cells/query)
VI = 0.0


class DriftOracle:
    def __init__(self):
        self.nrows = N
        self.ncols = N
        self.covs = np.zeros((1, N, N), dtype=np.float32)
        self.query_count = 0

    def in_bounds(self, i, j):
        return 0 <= i < N and 0 <= j < N

    def reset_counter(self):
        self.query_count = 0

    @property
    def total_cells(self):
        return N * N

    def query_cell(self, i, j):
        self.query_count += 1
        if not self.in_bounds(i, j):
            return None, None
        ti = 50.0 + VI * self.query_count
        tj = 8.0 + VJ * self.query_count
        pres = float(np.hypot(i - ti, j - tj)) <= R
        return bool(pres), self.covs[:, i, j].copy()


CFG = dict(target_points=10, max_queries=400, min_dist_cells=1.0,
           stage2_min=2, stage3_min=3, stage3_jump_interval=20,
           stage3_jump_min_ratio=1.25, stage3_jump_max_ratio=5.0,
           curiosity_weight=1.0, guidance_on=True, cov_in_state=False,
           sparse_guidance=False, stage1_grid_step=2,
           coverage_budget_ratio=0.05, multicluster=False, cov_indices=[0])

SEEDS = list(range(101, 111))


def run(motion):
    cfg = dict(CFG)
    cfg["motion_extrapolation"] = motion
    out = []
    for s in SEEDS:
        o = DriftOracle()
        r = SpeciesSearcher(o, cfg).run_episode(s)
        out.append(r["points_found"])
    return out


if __name__ == "__main__":
    off = run(False)
    on = run(True)
    mo, mn = float(np.mean(off)), float(np.mean(on))
    print(f"motion=off mean pts={mo:.2f}  per-seed={off}", flush=True)
    print(f"motion=on  mean pts={mn:.2f}  per-seed={on}", flush=True)
    print(f"Δ={mn-mo:+.2f} ({(mn-mo)/mo*100 if mo>0 else float('nan'):+.1f}%)", flush=True)
