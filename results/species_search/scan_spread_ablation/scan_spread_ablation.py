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
"""scan_spread_ablation.py — 改进1（冷启动盲扫铺开）验收。

合成"目标云列带"场景：120x120 网格，稀疏正样本集中在少数列带，
小预算(500)下对比 stage1_scan_mode=serpentine（旧蛇形光栅）vs spread（新铺开）。
agent 的起始格由 seed 决定（随机列），serpentine 只扫起始列附近几列，
spread 则把首批查询均匀撒到全列轴。seeds 101-120 × 20 对比 points_found。

用法: python -X utf8 scan_spread_ablation.py
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

# ---------------- 合成 oracle：目标云集中在少数列带 ----------------
NROWS = 120
NCOLS = 120
# 三条纵向目标列带（列区间），行方向散布稀疏正样本
BAND_COLS = [(18, 24), (58, 64), (96, 102)]
SEED_BUILD = 7


class BandOracle:
    """接口对齐 SpeciesOracle：nrows/ncols/covs/in_bounds/query_cell/reset_counter。"""

    def __init__(self):
        self.nrows = NROWS
        self.ncols = NCOLS
        rng = np.random.RandomState(SEED_BUILD)
        # 协变量：两个平滑场（Stage2 导向用，Stage1 盲扫不依赖）
        yy, xx = np.mgrid[0:NROWS, 0:NCOLS]
        cov0 = np.sin(xx / 18.0) + 0.3 * rng.randn(NROWS, NCOLS)
        cov1 = np.cos(yy / 22.0) + 0.3 * rng.randn(NROWS, NCOLS)
        self.covs = np.stack([cov0.astype(np.float32), cov1.astype(np.float32)])
        # presence：三条列带内稀疏撒点（带内密度较高，带外为 0）
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
    print("构建目标云列带 oracle：", BAND_COLS, flush=True)
    demo = BandOracle()
    n_pos = int(demo.presence.sum())
    print(f"  正样本总数={n_pos}  稀疏度={n_pos/(NROWS*NCOLS)*100:.2f}%", flush=True)

    print("\n跑 serpentine ...", flush=True)
    serp = run_mode("serpentine")
    print("跑 spread ...", flush=True)
    spread = run_mode("spread")

    sp = [r[1] for r in serp]
    dp = [r[1] for r in spread]
    m_sp = float(np.mean(sp))
    m_dp = float(np.mean(dp))
    rel = (m_dp - m_sp) / m_sp * 100.0 if m_sp > 0 else float("inf")
    hitrate_sp = sum(1 for x in sp if x > 0) / len(sp)
    hitrate_dp = sum(1 for x in dp if x > 0) / len(dp)

    print(f"\n=== serpentine: mean points_found={m_sp:.2f}  命中率={hitrate_sp*100:.0f}% ===", flush=True)
    print(f"=== spread    : mean points_found={m_dp:.2f}  命中率={hitrate_dp*100:.0f}% ===", flush=True)
    print(f"=== 相对提升 = {rel:+.1f}%   命中率比 = {hitrate_dp/max(hitrate_sp,1e-9):.2f}x ===", flush=True)

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

    # 图：箱线 + 种子序列折线
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].boxplot([sp, dp], labels=["serpentine", "spread"], showmeans=True)
    axes[0].set_ylabel("points_found")
    axes[0].set_title(f"盲扫模式对比（{NROWS}x{NCOLS}，预算500，seeds101-120）")
    axes[1].plot(SEEDS, sp, "o-", label=f"serpentine 均值{m_sp:.1f}")
    axes[1].plot(SEEDS, dp, "s-", label=f"spread 均值{m_dp:.1f}")
    axes[1].set_xlabel("seed")
    axes[1].set_ylabel("points_found")
    axes[1].set_title(f"逐种子命中（相对提升 {rel:+.0f}%）")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "scan_spread_ablation.png", dpi=130)
    print(f"\n[done] CSV+PNG -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
