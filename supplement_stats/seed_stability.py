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
"""seed_stability.py — 种子稳定性判断（任务16）。

证据链：
  E1 确定性（同 seed 复现）：results/chl_search/regress_v2_v0cfg/rng_crosscheck.csv
     —— 单进程 vs 多线程，8 seed 的 points/queries 全部一致。
  E2 跨种子分布（20 seeds）：MC framework_mc_detail（20 world × 20 algo）、
     species clouded_leopard / black_muntjac params19。
  E3 跨种子分布（10 seeds）：timevary v5 per_seed_*_decay300（moving_plume/abrupt_jump）。
输出：supplement_stats/seed_stability.csv + 控制台结论。
"""
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(REPO_ROOT)
OUT = HERE / "seed_stability.csv"


def load(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def stats_of(vals):
    a = np.array(vals, float)
    lo, hi = np.percentile(a, [2.5, 97.5])
    return dict(n=len(a), mean=round(a.mean(), 3), sd=round(a.std(ddof=1), 3),
                min=round(a.min(), 1), p25=round(np.percentile(a, 25), 2),
                median=round(np.median(a), 2), p75=round(np.percentile(a, 75), 2),
                max=round(a.max(), 1), ci95=f"{lo:.2f}-{hi:.2f}")


def main():
    out_rows = []

    # E1 确定性
    rng = load(ROOT / "results" / "chl_search" / "regress_v2_v0cfg" / "rng_crosscheck.csv")
    pm = sum(1 for r in rng if r["points_match"] == "True")
    qm = sum(1 for r in rng if r["queries_match"] == "True")
    print(f"[E1] RNG 确定性：{len(rng)} seed，points 一致 {pm}/{len(rng)}，queries 一致 {qm}/{len(rng)}")
    out_rows.append(dict(evidence="E1_rng_determinism", unit=f"chl_compare_{len(rng)}seeds",
                         metric="points+queries", n=len(rng),
                         mean=round(pm / max(len(rng), 1), 3), sd=0,
                         min=0, p25=0, median=round(qm / max(len(rng), 1), 3),
                         p75=0, max=0, ci95="match_rate"))

    # E2 MC：20 world × 20 algo seeds
    mc = load(ROOT / "results" / "minecraft_search" / "framework_mc_detail.csv")
    pts = stats_of([r["points_found"] for r in mc])
    qs = stats_of([r["total_queries"] for r in mc])
    # 按 world_seed 聚合：world 间稳定性（每 world 20 algo seed 的均值）
    by_w = {}
    for r in mc:
        by_w.setdefault(r["world_seed"], []).append(float(r["points_found"]))
    w_means = [np.mean(v) for v in by_w.values()]
    w_st = stats_of(w_means)
    print(f"[E2] MC framework：{pts['n']} runs pts mean={pts['mean']} sd={pts['sd']} "
          f"CI95={pts['ci95']} 四分位={pts['p25']}/{pts['median']}/{pts['p75']}")
    print(f"     world 均值分布：mean={w_st['mean']} sd={w_st['sd']} "
          f"range={w_st['min']}-{w_st['max']}（20 worlds）")
    out_rows.append(dict(evidence="E2_MC_400runs", unit="20w×20s", metric="points",
                         n=pts["n"], mean=pts["mean"], sd=pts["sd"],
                         min=pts["min"], p25=pts["p25"], median=pts["median"],
                         p75=pts["p75"], max=pts["max"], ci95=pts["ci95"]))
    out_rows.append(dict(evidence="E2_MC_world_mean", unit="20worlds", metric="points",
                         n=w_st["n"], mean=w_st["mean"], sd=w_st["sd"],
                         min=w_st["min"], p25=w_st["p25"], median=w_st["median"],
                         p75=w_st["p75"], max=w_st["max"], ci95=w_st["ci95"]))

    # E2b species 20 seeds
    for sp, tag in [("clouded_leopard", "species_clouded_leopard"),
                    ("black_muntjac", "species_black_muntjac")]:
        p = ROOT / "results" / "species_search" / f"framework_species_{sp}_params19.csv"
        if not p.exists():
            print(f"[skip] {p.name} 不存在")
            continue
        sub = load(p)
        pts = stats_of([r["points_found"] for r in sub])
        qs = stats_of([r["total_queries"] for r in sub])
        print(f"[E2b] {tag} params19：n={pts['n']} pts mean={pts['mean']} sd={pts['sd']} "
              f"CI95={pts['ci95']}")
        out_rows.append(dict(evidence=f"E2b_{tag}", unit="20seeds", metric="points",
                             n=pts["n"], mean=pts["mean"], sd=pts["sd"],
                             min=pts["min"], p25=pts["p25"], median=pts["median"],
                             p75=pts["p75"], max=pts["max"], ci95=pts["ci95"]))

    # E3 timevary 10 seeds
    for sc, tag in [("moving_plume", "tv_moving_plume"), ("abrupt_jump", "tv_abrupt_jump")]:
        p = ROOT / "results" / "timevary_search" / "v5" / f"per_seed_{sc}_framework_decay300.csv"
        if not p.exists():
            print(f"[skip] {p.name} 不存在")
            continue
        sub = load(p)
        pts = stats_of([r["points_found"] for r in sub])
        qs = stats_of([r["total_queries"] for r in sub])
        print(f"[E3] {tag} decay300：n={pts['n']} pts mean={pts['mean']} sd={pts['sd']} "
              f"CI95={pts['ci95']} q mean={qs['mean']} sd={qs['sd']}")
        out_rows.append(dict(evidence=f"E3_{tag}", unit="10seeds", metric="points",
                             n=pts["n"], mean=pts["mean"], sd=pts["sd"],
                             min=pts["min"], p25=pts["p25"], median=pts["median"],
                             p75=pts["p75"], max=pts["max"], ci95=pts["ci95"]))

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print(f"[done] 输出 -> {OUT}")


if __name__ == "__main__":
    main()
