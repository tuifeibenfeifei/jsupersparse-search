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
"""scan_spread_ablation.py — Improvement 1 (cold-start blind sweep spread) validation.

Synthetic "target cloud column bands" scenario: 120x120 grid, sparse positives concentrated in a few column bands,
Under small budget (500), compare stage1_scan_mode=serpentine (old serpentine raster) vs spread (new spread).
Agent's starting cell is determined by seed (random column); serpentine only scans a few columns near the start,
while spread distributes the first batch of queries evenly across the full column axis. seeds 101-120 × 20 compare points_found.

Usage: python -X utf8 scan_spread_ablation.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(REPO_ROOT)
sys.path.insert(0, str(ROOT / "species-search"))
from species_agent import SpeciesSearcher  # noqa: E402

OUT = ROOT / "results" / "species_search" / "scan_spread_ablation"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------- Synthetic oracle: target cloud concentrated in a few column bands ----------------
NROWS = 120
NCOLS = 120
# Three vertical target column bands (column ranges), sparse positives scattered in row direction
BAND_COLS = [(18, 24), (58, 64), (96, 102)]
SEED_BUILD = 7


class BandOracle:
    """Interface aligned with SpeciesOracle: nrows/ncols/covs/in_bounds/query_cell/reset_counter."""

    def __init__(self):
        self.nrows = NROWS
        self.ncols = NCOLS
        rng = np.random.RandomState(SEED_BUILD)
        # Covariates: two smooth fields (used for Stage2 guidance; Stage1 blind sweep does not depend on them)
        yy, xx = np.mgrid[0:NROWS, 0:NCOLS]
        cov0 = np.sin(xx / 18.0) + 0.3 * rng.randn(NROWS, NCOLS)
        cov1 = np.cos(yy / 22.0) + 0.3 * rng.randn(NROWS, NCOLS)
        self.covs = np.stack([cov0.astype(np.float32), cov1.astype(np.float32)])
        # presence: sparse points within three column bands (higher density inside bands, 0 outside)
        self.presence = np.zeros((NROWS, NCOLS), dtype=bool)
        for (c0, c1) in BAND_COLS:
            for j in range(c0, c1 + 1):
                rows = rng.choice(np.arange(8, NROWS - 8), size=5, replace=False)
                for i in rows:
                    self.presence[i, j] = True
        self.query_count = 0

    @property
    def total_cells(self):
        return self.nrows * self.ncols

    def in_bounds(self, i, j):
        return 0 <= i < self.nrows and 0 <= j < self.ncols

    def reset_counter(self):
        self.query_count = 0

    def query_cell(self, i, j):
        self.query_count += 1
        if not self.in_bounds(i, j):
            return None, None
        return bool(self.presence[i, j]), self.covs[:, i, j].copy()


CFG = dict(target_points=20, max_queries=500, min_dist_cells=3.0,
           stage2_min=3, stage3_min=5, stage3_jump_interval=30,
           stage3_jump_min_ratio=1.25, stage3_jump_max_ratio=5.0,
           curiosity_weight=1.0, guidance_on=True, cov_in_state=True,
           recalc_every=20, stage2_topk=8, sparse_guidance=False,
           stage1_grid_step=2, coverage_budget_ratio=0.08,
           multicluster=False, cov_indices=[0, 1])

SEEDS = list(range(101, 121))


def run_mode(mode):
    cfg = dict(CFG)
    cfg["stage1_scan_mode"] = mode
    rows = []
    for seed in SEEDS:
        orc = BandOracle()
        res = SpeciesSearcher(orc, cfg).run_episode(seed)
        rows.append([seed, res["points_found"], res["total_queries"],
                     res["covered_cells"], res["final_stage"]])
    return rows


def main():
    print("Building target cloud column band oracle:", BAND_COLS, flush=True)
    demo = BandOracle()
    n_pos = int(demo.presence.sum())
    print(f"  Total positives={n_pos}  Sparsity={n_pos/(NROWS*NCOLS)*100:.2f}%", flush=True)

    print("\nRunning serpentine ...", flush=True)
    serp = run_mode("serpentine")
    print("Running spread ...", flush=True)
    spread = run_mode("spread")

    sp = [r[1] for r in serp]
    dp = [r[1] for r in spread]
    m_sp = float(np.mean(sp))
    m_dp = float(np.mean(dp))
    rel = (m_dp - m_sp) / m_sp * 100.0 if m_sp > 0 else float("inf")
    hitrate_sp = sum(1 for x in sp if x > 0) / len(sp)
    hitrate_dp = sum(1 for x in dp if x > 0) / len(dp)

    print(f"\n=== serpentine: mean points_found={m_sp:.2f}  hit rate={hitrate_sp*100:.0f}% ===", flush=True)
    print(f"=== spread    : mean points_found={m_dp:.2f}  hit rate={hitrate_dp*100:.0f}% ===", flush=True)
    print(f"=== Relative improvement = {rel:+.1f}%   hit rate ratio = {hitrate_dp/max(hitrate_sp,1e-9):.2f}x ===", flush=True)

    # CSV
    with open(OUT / "scan_spread_ablation.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["seed", "serpentine_pts", "serpentine_q", "serpentine_cov",
                    "serpentine_stage", "spread_pts", "spread_q", "spread_cov", "spread_stage"])
        for s_row, d_row in zip(serp, spread):
            w.writerow([s_row[0], s_row[1], s_row[2], s_row[3], s_row[4],
                        d_row[1], d_row[2], d_row[3], d_row[4]])
        w.writerow(["MEAN", round(m_sp, 2), "", "", "", round(m_dp, 2), "", "", ""])
        w.writerow(["HIT_RATE", f"{hitrate_sp:.2f}", "", "", "", f"{hitrate_dp:.2f}", "", "", ""])
        w.writerow(["REL_IMPROVE_PCT", f"{rel:.1f}", "", "", "", "", "", "", ""])

    # Figure: boxplot + seed sequence line plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].boxplot([sp, dp], labels=["serpentine", "spread"], showmeans=True)
    axes[0].set_ylabel("points_found")
    axes[0].set_title(f"Blind sweep mode comparison ({NROWS}x{NCOLS}, budget 500, seeds 101-120)")
    axes[1].plot(SEEDS, sp, "o-", label=f"serpentine mean {m_sp:.1f}")
    axes[1].plot(SEEDS, dp, "s-", label=f"spread mean {m_dp:.1f}")
    axes[1].set_xlabel("seed")
    axes[1].set_ylabel("points_found")
    axes[1].set_title(f"Per-seed hits (relative improvement {rel:+.0f}%)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "scan_spread_ablation.png", dpi=130)
    print(f"\n[done] CSV+PNG -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
