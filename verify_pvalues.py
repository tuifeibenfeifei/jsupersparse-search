# -*- coding: utf-8 -*-
"""verify_pvalues.py - independently recompute the paired-test p values of the
paper tables.

Uses only the detail CSVs already committed to the repository, with the same
protocol the paper declares:
- Minecraft: world-level means (average over seeds within each world), then a
  paired one-sided test over the 20 worlds.
- All other scenarios: seed-level pairing.

The output is printed next to the paper values so each line can be judged on
its own.
"""
from __future__ import annotations

import csv
import io
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent


def read_csv(rel):
    with io.open(ROOT / rel, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def mc_world_means():
    """{(method): {world: mean_points}} and {world: mean_queries} for solved worlds."""
    out = {}
    fw = read_csv("results/minecraft_search/framework_mc_detail.csv")
    d = defaultdict(list)
    q = defaultdict(list)
    for r in fw:
        w = int(r["world_seed"])
        d[w].append(float(r["points_found"]))
        q[w].append(float(r["total_queries"]))
    out["framework"] = {w: np.mean(v) for w, v in d.items()}
    out["framework_q"] = {w: np.mean(v) for w, v in q.items()}

    for name, fn in [("random", "baseline_mc_random.csv"),
                     ("hill_climb", "baseline_mc_hill_climb.csv"),
                     ("grid", "baseline_mc_grid.csv")]:
        rows = read_csv("results/minecraft_search/" + fn)
        d = defaultdict(list)
        q = defaultdict(list)
        for r in rows:
            w = int(r["world_seed"])
            d[w].append(float(r["points_found"]))
            q[w].append(float(r["total_queries"]))
        out[name] = {w: np.mean(v) for w, v in d.items()}
        out[name + "_q"] = {w: np.mean(v) for w, v in q.items()}
    return out


def paired_p(a, b, alternative="greater"):
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    try:
        w = stats.wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
        pw = w.pvalue
    except Exception as e:
        pw = float("nan")
    try:
        t = stats.ttest_rel(a, b, alternative=alternative)
        pt = t.pvalue
    except Exception:
        pt = float("nan")
    return pw, pt, diff.mean()


def main():
    m = mc_world_means()
    worlds = sorted(m["framework"])
    print("=" * 78)
    print("Table 2 / Table 10 — Minecraft world-level paired tests (n = %d worlds)" % len(worlds))
    print("=" * 78)
    print(f"{'comparison':34} {'meanA':>8} {'meanB':>8} {'diff':>8} {'wilcoxon_p':>11} {'ttest_p':>9}")
    fw = [m["framework"][w] for w in worlds]
    for name in ("random", "hill_climb", "grid"):
        b = [m[name][w] for w in worlds]
        pw, pt, d = paired_p(fw, b)
        print(f"{'framework vs ' + name:34} {np.mean(fw):8.3f} {np.mean(b):8.3f} {d:+8.3f} {pw:11.6f} {pt:9.6f}")

    # queries on solvable worlds
    solv = [w for w in worlds if m["framework"][w] > 0 and m["random"][w] > 0]
    print()
    print(f"queries, solvable worlds only (n = {len(solv)}):")
    a = [m["framework_q"][w] for w in solv]
    b = [m["random_q"][w] for w in solv]
    pw, pt, d = paired_p(a, b, alternative="less")
    print(f"{'framework vs random (queries)':34} {np.mean(a):8.2f} {np.mean(b):8.2f} {d:+8.2f} {pw:11.6f} {pt:9.6f}")

    print()
    print("paper says: 0.055 (hits vs random), 0.003 (vs hill climb), 0.001 (vs grid), 0.034 (queries)")


if __name__ == "__main__":
    main()
