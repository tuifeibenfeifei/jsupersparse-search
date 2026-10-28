# -*- coding: utf-8 -*-
import os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve()
for _ in range(8):
    if (REPO_ROOT / 'configs').is_dir():
        break
    REPO_ROOT = REPO_ROOT.parent
"""Minecraft paired-test recomputation (the first four rows of TABLE XI in the paper).

With n=20, the one-sided exact Wilcoxon cannot produce p=0.055; this script uses the per-run data under results/minecraft_search/
to recompute the four comparisons under multiple test specifications, in order to clarify or correct the paper statements.
Data: framework_mc_detail.csv (20 worlds x 20 algo seeds), baseline_mc_random / hill_climb / grid (20 worlds x 5 seeds).
Specification: world-level mean -> 20 pairs.
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats

R = Path(REPO_ROOT / "results/minecraft_search")


def load(fn):
    with open(R / fn, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    d = {}
    for r in rows:
        ws = int(r["world_seed"])
        d.setdefault(ws, []).append((float(r["points_found"]), float(r["total_queries"])))
    return {ws: (np.mean([p for p, q in v]), np.mean([q for p, q in v])) for ws, v in d.items()}


def report(name, a_pts, b_pts, a_q, b_q, expect=None):
    print("=" * 72)
    print(f"[{name}]  n={len(a_pts)}  mean_a={np.mean(a_pts):.3f}  mean_b={np.mean(b_pts):.3f}  "
          f"diff={np.mean(a_pts)-np.mean(b_pts):+.3f}")
    if expect:
        print(f"    paper value: {expect}")
    d = np.asarray(a_pts) - np.asarray(b_pts)  # a - b
    for method in ("auto", "exact", "approx"):
        for alt in ("greater", "less", "two-sided"):
            try:
                p = stats.wilcoxon(d, zero_method="wilcox", alternative=alt, method=method).pvalue
                print(f"    wilcoxon method={method:6s} alt={alt:9s} -> p={p:.6f}")
            except Exception as e:
                print(f"    wilcoxon method={method:6s} alt={alt:9s} -> ERR {e}")
    # paired t
    t = stats.ttest_rel(a_pts, b_pts)
    print(f"    paired t: p_two={t.pvalue:.6f}  p_one(greater)={t.pvalue/2 if t.statistic>0 else 1-t.pvalue/2:.6f}")
    # sign test (binomial, exact)
    pos = int(np.sum(d > 0)); neg = int(np.sum(d < 0))
    if pos + neg > 0:
        sp = stats.binomtest(pos, pos + neg, 0.5, alternative="greater").pvalue
        print(f"    sign test (exact): pos={pos} neg={neg} p(greater)={sp:.6f}")
    # cliff's delta
    cd = (sum(np.sum(np.asarray(b_pts) < x) - np.sum(np.asarray(b_pts) > x) for x in a_pts)
          / (len(a_pts) * len(b_pts)))
    print(f"    cliff's delta={cd:+.4f}")


fw = load("framework_mc_detail.csv")
rnd = load("baseline_mc_random.csv")
hill = load("baseline_mc_hill_climb.csv")
grid = load("baseline_mc_grid.csv")

worlds = sorted(set(fw) & set(rnd) & set(hill) & set(grid))

# 1) vs random, hits
report("framework vs random (hits)",
       [fw[w][0] for w in worlds], [rnd[w][0] for w in worlds], None, None,
       "p = 0.055, delta = +0.007, diff = +0.415")

# 2) vs hill climbing, hits
report("framework vs hill_climb (hits)",
       [fw[w][0] for w in worlds], [hill[w][0] for w in worlds], None, None,
       "p = 0.003, diff = +8.405")

# 3) vs grid, hits
report("framework vs grid (hits)",
       [fw[w][0] for w in worlds], [grid[w][0] for w in worlds], None, None,
       "p = 0.001, diff = +16.115")

# 4) vs random, queries on attainable subset (loosest definition: worlds where the framework world-mean hits reach 40)
attain = [w for w in worlds if abs(fw[w][0] - 40) < 1e-9]
print("=" * 72)
print(f"[attainable subset: framework world-mean hits == 40] worlds={attain}")
if attain:
    aq = [fw[w][1] for w in attain]
    bq = [rnd[w][1] for w in attain]
    report("framework vs random (queries, attainable subset)", aq, bq, None, None,
           "p = 0.034, diff = -227.0 (622.9 vs 850.0)")
