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
"""motion_extrap_ablation.py — 改进3（运动外推，默认关闭）A/B 自测。

海上搜救场景：MaritimeOracle 的 presence 沿 traj 随 query_count(时间) 漂移，
静态栖息地假设失效。对比 motion_extrapolation=False（基线，variant F 行为）
vs True（开运动外推），seeds 101-110。单进程串行。

用法: python -X utf8 motion_extrap_ablation.py
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(REPO_ROOT)
sys.path.insert(0, str(ROOT / "species-search"))
sys.path.insert(0, str(ROOT / "maritime-search"))

from species_agent import SpeciesSearcher          # noqa: E402
from build_maritime_oracle import MaritimeOracle   # noqa: E402

OUT = ROOT / "results" / "maritime_search" / "motion_extrap_ablation"
OUT.mkdir(parents=True, exist_ok=True)
NPZ = ROOT / "maritime-search" / "data" / "oracle.npz"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

SEEDS = list(range(101, 111))

# 与 run_maritime.py 同口径 config
BASE = dict(target_points=15, max_queries=600, min_dist_cells=1.0,
            stage1_grid_step=4, cov_indices=list(range(19)),
            multicluster=True, cluster_ring_ratio=10.0, cluster_idle_limit=12,
            blank_jump=True, coverage_budget_ratio=0.12, blank_far_ratio=0.3,
            post_jump_grace=6, saturate_mask=False, idle_blank_mix=8,
            sparse_guidance=True, sparse_topk=6, sparse_min_pos=5, recalc_every=20,
            stage2_min=3, stage3_min=5, stage3_jump_interval=30)


def run_mode(motion):
    cfg = dict(BASE)
    cfg["motion_extrapolation"] = motion
    rows = []
    for seed in SEEDS:
        orc = MaritimeOracle(NPZ)   # 每 seed fresh oracle（query_count 由 run_episode 复位）
        t0 = time.time()
        res = SpeciesSearcher(orc, cfg).run_episode(seed)
        rows.append([seed, res["points_found"], res["total_queries"],
                     res["covered_cells"], res["final_stage"]])
        print(f"  motion={motion} seed={seed} pts={res['points_found']} "
              f"q={res['total_queries']} stage={res['final_stage']} "
              f"({time.time()-t0:.1f}s)", flush=True)
    return rows


def main():
    print("=== motion_extrapolation=False（基线）===", flush=True)
    off = run_mode(False)
    print("=== motion_extrapolation=True（开外推）===", flush=True)
    on = run_mode(True)

    off_p = [r[1] for r in off]
    on_p = [r[1] for r in on]
    m_off, m_on = float(np.mean(off_p)), float(np.mean(on_p))
    delta = m_on - m_off
    print(f"\n=== 基线 mean pts={m_off:.2f}   开外推 mean pts={m_on:.2f}   "
          f"Δ={delta:+.2f} ({(delta/m_off*100 if m_off>0 else float('nan')):+.1f}%) ===", flush=True)

    with open(OUT / "motion_extrap_ablation.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["seed", "off_pts", "off_q", "off_cov", "off_stage",
                    "on_pts", "on_q", "on_cov", "on_stage"])
        for o_row, n_row in zip(off, on):
            w.writerow([o_row[0], o_row[1], o_row[2], o_row[3], o_row[4],
                        n_row[1], n_row[2], n_row[3], n_row[4]])
        w.writerow(["MEAN", round(m_off, 2), "", "", "", round(m_on, 2), "", "", ""])
        w.writerow(["DELTA", round(delta, 2), "", "", "", "", "", "", ""])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(SEEDS))
    ax.bar(x - 0.2, off_p, 0.4, label=f"motion=off 均值{m_off:.1f}")
    ax.bar(x + 0.2, on_p, 0.4, label=f"motion=on 均值{m_on:.1f}")
    ax.set_xticks(x)
    ax.set_xticklabels(SEEDS)
    ax.set_xlabel("seed")
    ax.set_ylabel("points_found")
    ax.set_title("海上搜救：运动外推开/关 A/B（动态漂移 oracle）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "motion_extrap_ablation.png", dpi=130)
    print(f"[done] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
