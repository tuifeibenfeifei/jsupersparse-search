# -*- coding: utf-8 -*-
"""Unified statistics presentation: recompute from existing CSVs (do not rerun experiments).

Inputs ablation_detail.csv (3 scenarios x full/AUC_off/multicluster_off) and screen_mi_vs_auc.csv;
Outputs peerj_stats_unified.csv (unified stats long table), also prints Wilson CI and anchor self-check.
"""
import csv
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(r"C:\Users\19264\Desktop\deep-sea-exploration")
ABL = ROOT / "results" / "ablation_cross" / "ablation_detail.csv"
SCR = ROOT / "supplement_stats" / "screen_mi_vs_auc.csv"
OUT_CSV = ROOT / "supplement_stats" / "peerj_stats_unified.csv"

ALGO_SEEDS = list(range(101, 121))   # 101..120


# Read ablation
def load_ablation():
    rows = []
    with open(ABL, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "scenario": r["scenario"],
                "variant": r["variant"],
                "world_seed": int(r["world_seed"]),
                "algo_seed": int(r["algo_seed"]),
                "points_found": float(r["points_found"]),
                "total_queries": float(r["total_queries"]),
            })
    return rows


def cliff_delta(a, b):
    """Cliff's delta on paired diff d=a-b: P(d>0)-P(d<0) over non-ties."""
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    n_pos = int(np.sum(d > 0))
    n_neg = int(np.sum(d < 0))
    n_nonzero = n_pos + n_neg
    if n_nonzero == 0:
        return 0.0
    return (n_pos - n_neg) / n_nonzero


def paired_compare(rows, scenario, variant, metric):
    """Paired: variant - full; MC paired by (world, algo), others by algo_seed."""
    key = ("world_seed", "algo_seed") if scenario == "minecraft" else ("algo_seed",)
    full = {}
    var = {}
    for r in rows:
        if r["scenario"] != scenario:
            continue
        k = tuple(r[k_] for k_ in key)
        if r["variant"] == "full":
            full[k] = r[metric]
        elif r["variant"] == variant:
            var[k] = r[metric]
    keys = sorted(set(full) & set(var))
    d = np.array([var[k] - full[k] for k in keys], dtype=float)
    return d, keys


def one_sided_wilcoxon(d, alternative):
    """Paired Wilcoxon signed-rank (one-tailed); all-zero diff returns p=1.0."""
    if np.all(d == 0):
        return 1.0
    try:
        res = stats.wilcoxon(d, alternative=alternative, zero_method="wilcox")
        return float(res.pvalue)
    except ValueError:
        return float("nan")


# Wilson CI
def wilson_ci(phat, n, z=1.959963984540054):
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


# main
def main():
    rows = load_ablation()

    # Anchor self-check
    print("=== Anchor self-check (chlorophyll 20-seed mean total_queries) ===")
    for v, expect in [("full", 1596.25), ("AUC_off", 1805.0), ("multicluster_off", 958.0)]:
        vals = [r["total_queries"] for r in rows if r["scenario"] == "chlorophyll" and r["variant"] == v]
        m = float(np.mean(vals))
        ok = abs(m - expect) <= 5.0
        print(f"  chlorophyll/{v:16s} mean={m:8.2f}  expect~{expect:8.2f}  n={len(vals)}  {'OK' if ok else 'MISMATCH'}")

    # Unified table: each scenario x {AUC_off, multicluster_off} x {total_queries, points_found}
    # One-tailed direction (consistent with harmful_arms_paired.csv pre-registered hypotheses):
    #   AUC_off        : disabling guidance should make queries more (greater), hits fewer (less)
    #   multicluster_off: disabling multicluster should make queries fewer (less, H2), hits fewer (less)
    spec = {
        "AUC_off":          {"total_queries": "greater", "points_found": "less"},
        "multicluster_off":  {"total_queries": "less",    "points_found": "less"},
    }
    out = []
    print("\n=== Unified comparison table (delta = variant - full) ===")
    hdr = ["scenario", "variant", "metric", "n_pairs",
           "delta_mean", "delta_sd", "wilcoxon_p_onesided", "cliff_delta",
           "direction", "sig_note"]
    print("  " + " | ".join(hdr))
    for scenario in ["chlorophyll", "clouded_leopard", "minecraft"]:
        for variant in ["AUC_off", "multicluster_off"]:
            for metric in ["total_queries", "points_found"]:
                d, keys = paired_compare(rows, scenario, variant, metric)
                alt = spec[variant][metric]
                p = one_sided_wilcoxon(d, alt)
                cd = cliff_delta(d, np.zeros_like(d))
                dm, ds = float(np.mean(d)), float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
                if dm > 0:
                    direction = "variant>full(off more)"
                elif dm < 0:
                    direction = "variant<full(off fewer)"
                else:
                    direction = "no difference"
                if np.isnan(p):
                    note = "cannot test (all identical)"
                elif p < 0.05:
                    note = "p<0.05 significant"
                else:
                    note = "n.s. (p>=0.05, direction hypothesis only)"
                out.append([scenario, variant, metric, len(keys),
                            round(dm, 3), round(ds, 3), p, round(cd, 4),
                            direction, note])
                print(f"  {scenario:16s} {variant:18s} {metric:14s} n={len(keys):3d} "
                      f"Δ={dm:9.2f}±{ds:7.2f} p={p:.4g} δ={cd:+.3f} {note}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(out)
    print(f"\n[saved] {OUT_CSV}  ({len(out)} rows)")

    # Wilson
    print("\n=== Wilson 95% CI for selector precision 0.763 ===")
    scr_rows = []
    with open(SCR, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["sel_true"] == "":
                continue
            scr_rows.append(r)
    print(f"  Valid per-event rows (sel_true non-empty) = {len(scr_rows)}")
    for screen in ["auc", "mi"]:
        sub = [r for r in scr_rows if r["screen"] == screen]
        tp = sum(int(r["sel_true"]) for r in sub)
        fp = sum(int(r["sel_noise"]) for r in sub)
        # Per-event retained count = sel_true+sel_noise (observed as 6 or 7), pooled sum
        n = tp + fp
        phat = tp / n
        lo, hi = wilson_ci(phat, n)
        print(f"  [screen={screen:3s}] events={len(sub)} "
              f"TP(true signal retained)={tp}  FP(noise retained)={fp}  n={n}  "
              f"p̂={phat:.4f}  Wilson95%CI=[{lo:.4f},{hi:.4f}]")
        print(f"        Chance level 7/19={7/19:.4f}; outside CI: {not (lo <= 7/19 <= hi)}")


if __name__ == "__main__":
    main()
