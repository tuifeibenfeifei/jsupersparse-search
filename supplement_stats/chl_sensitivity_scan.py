# -*- coding: utf-8 -*-
import os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve()
for _ in range(8):
    if (REPO_ROOT / 'configs').is_dir():
        break
    REPO_ROOT = REPO_ROOT.parent
"""Supplementary run of the TABLE XIV sensitivity scans (seeds 101-110, n=10).

coverage_budget_ratio: 0.05 / 0.08 / 0.12(default) / 0.20 / 0.30;
weak_signal_min_auc: 0.50 / 0.53 / 0.52(default) / 0.55 / 0.60 / 0.70.
Output: supplement_stats/chl_sensitivity_scan.csv (mean queries + change vs default).
"""
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(REPO_ROOT)
sys.path.insert(0, str(ROOT / "chlorophyll-search"))
sys.path.insert(0, str(ROOT))

from oracle import SpeciesOracle  # noqa: E402
from species_agent import SpeciesSearcher  # noqa: E402

OUT = ROOT / "supplement_stats" / "chl_sensitivity_scan.csv"
LOG = ROOT / "supplement_stats" / "chl_sensitivity_scan.log"
ORACLE_PATH = ROOT / "chlorophyll-search" / "data" / "oracle.npz"
SEEDS = list(range(101, 111))  # matches the TABLE XIV declaration (n=10)

CHL_BASE = dict(
    target_points=20, max_queries=4000, min_dist_cells=1.0,
    stage2_min=3, stage3_min=5, stage3_jump_interval=30,
    stage3_jump_min_ratio=1.25, stage3_jump_max_ratio=5.0,
    curiosity_weight=1.0, learning_rate=0.1, discount_factor=0.95,
    epsilon_start=0.9, epsilon_end=0.1, epsilon_decay=0.999,
    stage1_grid_step=3, guidance_on=True, cov_in_state=True,
    recalc_every=20, stage2_topk=8,
    sparse_guidance=True, sparse_topk=6, sparse_min_pos=5,
    multicluster=True, cluster_ring_ratio=10.0, cluster_idle_limit=12,
    blank_jump=True, coverage_budget_ratio=0.12,
    blank_far_ratio=0.3, post_jump_grace=6, saturate_mask=False,
    cov_indices=list(range(19)), adaptive=True, directed_jump=True,
)

COVERAGE_VALUES = [0.05, 0.08, 0.12, 0.20, 0.30]
AUC_VALUES = [0.50, 0.52, 0.53, 0.55, 0.60, 0.70]


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cfg, seeds):
    oracle = SpeciesOracle(ORACLE_PATH)
    qs = []
    for s in seeds:
        res = SpeciesSearcher(oracle, cfg).run_episode(s)
        qs.append(res["total_queries"])
    return qs


def main():
    if LOG.exists():
        LOG.unlink()
    oracle = SpeciesOracle(ORACLE_PATH)
    t0 = time.time()

    # default row (shipped: coverage 0.12, auc 0.52)
    default_q = run(dict(CHL_BASE), SEEDS)
    def_mean = sum(default_q) / len(default_q)
    log(f"default (cov=0.12, auc=0.52) mean={def_mean:.1f} (n={len(default_q)})")

    rows = []
    rows.append(dict(param="default", value="0.12 / 0.52", mean_q=round(def_mean, 1),
                     change_pct="—", n=len(default_q)))

    # coverage_budget_ratio scan (auc held at 0.52)
    for v in COVERAGE_VALUES:
        if v == 0.12:
            continue
        cfg = dict(CHL_BASE, coverage_budget_ratio=v)
        qs = run(cfg, SEEDS)
        m = sum(qs) / len(qs)
        chg = (m - def_mean) / def_mean * 100.0
        rows.append(dict(param="coverage_budget_ratio", value=str(v),
                         mean_q=round(m, 1), change_pct=round(chg, 1), n=len(qs)))
        log(f"coverage_budget_ratio={v} mean={m:.1f} chg={chg:+.1f}%")

    # weak_signal_min_auc scan (coverage held at 0.12)
    for v in AUC_VALUES:
        if v == 0.52:
            continue
        cfg = dict(CHL_BASE, weak_signal_min_auc=v)
        qs = run(cfg, SEEDS)
        m = sum(qs) / len(qs)
        chg = (m - def_mean) / def_mean * 100.0
        rows.append(dict(param="weak_signal_min_auc", value=str(v),
                         mean_q=round(m, 1), change_pct=round(chg, 1), n=len(qs)))
        log(f"weak_signal_min_auc={v} mean={m:.1f} chg={chg:+.1f}%")

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["param", "value", "mean_q", "change_pct", "n"])
        w.writeheader()
        w.writerows(rows)
    log(f"[done] -> {OUT} ({len(rows)} rows, {(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
