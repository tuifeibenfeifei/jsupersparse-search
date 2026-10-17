# -*- coding: utf-8 -*-
import os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve()
for _ in range(8):
    if (REPO_ROOT / 'configs').is_dir():
        break
    REPO_ROOT = REPO_ROOT.parent
"""全量统一替换所需的剩余统计量（一次性算清）。

输入 ablation_detail.csv（full/AUC_off/multicluster_off）与 screen_off_arms.csv；
输出 final_stats.csv：chl/cl 各四组配对（queries/points，单尾 p + Cliff's δ，chl screen-off 另加 bootstrap 95% CI）。
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(REPO_ROOT)
ABL = ROOT / "results" / "ablation_cross" / "ablation_detail.csv"
SOF = ROOT / "supplement_stats" / "screen_off_arms.csv"
OUT = ROOT / "supplement_stats" / "final_stats.csv"


def load_abl(scene, variant, metric):
    vals = {}
    with open(ABL, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["scenario"] == scene and r["variant"] == variant:
                vals[int(r["algo_seed"])] = float(r["points_found" if metric == "points" else "total_queries"])
    return vals


def load_sof(scene, metric):
    vals = {}
    with open(SOF, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["scenario"] == scene and r["variant"] == "screen_off":
                vals[int(r["algo_seed"])] = float(r["points_found" if metric == "points" else "total_queries"])
    return vals


def cliff_delta(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    sgn = sum(np.sum(b < x) - np.sum(b > x) for x in a)
    return sgn / (len(a) * len(b))


def bootstrap_ci(d, n_boot=10000, seed=20261013):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    boots = [np.mean(rng.choice(d, size=len(d), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def analyze(scene, other, metric):
    full = load_abl(scene, "full", metric)
    if other == "screen_off":
        alt = load_sof(scene, metric)
    else:
        alt = load_abl(scene, other, metric)
    idx = sorted(set(full) & set(alt))
    a = [full[i] for i in idx]   # a = full
    b = [alt[i] for i in idx]    # b = 对照臂
    d = np.asarray(a) - np.asarray(b)
    p_gt = stats.wilcoxon(d, zero_method="wilcox", alternative="greater").pvalue
    p_lt = stats.wilcoxon(d, zero_method="wilcox", alternative="less").pvalue
    cd = cliff_delta(a, b)
    lo, hi = bootstrap_ci(d)
    print(f"[{scene}] full vs {other} ({metric}): n={len(idx)} "
          f"full={np.mean(a):.3f} other={np.mean(b):.3f} "
          f"diff(full-other)={np.mean(d):+.3f} "
          f"p(full>other)={p_gt:.4f} p(full<other)={p_lt:.4f} delta={cd:+.4f} "
          f"CI95(diff)=[{lo:+.2f}, {hi:+.2f}]")
    return dict(scene=scene, comparison=f"full_vs_{other}", metric=metric, n=len(idx),
                mean_full=round(float(np.mean(a)), 3), mean_other=round(float(np.mean(b)), 3),
                diff=round(float(np.mean(d)), 3), p_gt=round(float(p_gt), 4),
                p_lt=round(float(p_lt), 4), cliff_delta=round(float(cd), 4),
                ci_lo=round(lo, 2), ci_hi=round(hi, 2))


rows = []
rows.append(analyze("chlorophyll", "AUC_off", "queries"))      # = guidance-off entirely
rows.append(analyze("chlorophyll", "screen_off", "queries"))
rows.append(analyze("chlorophyll", "multicluster_off", "queries"))
rows.append(analyze("clouded_leopard", "AUC_off", "points"))   # = guidance-off entirely
rows.append(analyze("clouded_leopard", "screen_off", "points"))
rows.append(analyze("clouded_leopard", "multicluster_off", "points"))

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("WROTE", OUT)
