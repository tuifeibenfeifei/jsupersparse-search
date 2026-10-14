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
"""metric_robustness.py — 指标稳健性重排（任务15）。

检查：把"命中点数 points"换成"查询数 queries / 覆盖格数"作为排序指标，
方法排名是否翻转、框架的结论（负结果/有害臂）是否稳健。

数据源：
  S1 MC 深海：framework vs grid / hill_climb / random（baseline_mc_*.csv）
  S2 跨场景消融：3 场景 × full/AUC_off/multicluster_off（ablation_detail）
  S3 叶绿素全对比：14 方法（chl compare_detail）
输出：supplement_stats/metric_robustness.csv + 控制台结论。
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = Path(REPO_ROOT)
OUT = HERE / "metric_robustness.csv"


def load(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def summarize(tag, source, rows, metric_a, metric_b):
    by = {}
    for r in rows:
        k = r.get(source, "")
        by.setdefault(k, []).append(r)
    items = []
    for k, sub in by.items():
        va = np.array([float(r[metric_a]) for r in sub])
        vb = np.array([float(r[metric_b]) for r in sub])
        items.append(dict(tag=tag, arm=k, n=len(sub),
                          mean_a=round(va.mean(), 2), mean_b=round(vb.mean(), 1),
                          rank_a=0, rank_b=0))
    # 排名：A 指标大者为优（负结果视角：points 越大越好）
    items.sort(key=lambda x: (-x["mean_a"], x["arm"]))
    for i, it in enumerate(items, 1):
        it["rank_a"] = i
    items.sort(key=lambda x: (-x["mean_b"], x["arm"]))
    for i, it in enumerate(items, 1):
        it["rank_b"] = i
    items.sort(key=lambda x: x["arm"])
    ranks_a = [it["rank_a"] for it in items]
    ranks_b = [it["rank_b"] for it in items]
    rho, p = stats.spearmanr(ranks_a, ranks_b)
    flipped = [it for it in items if it["rank_a"] != it["rank_b"]]
    print(f"[{tag}] n_arms={len(items)} Spearman(rankA,rankB)={rho:.3f} p={p:.3f} "
          f"翻转臂={len(flipped)}")
    for it in items:
        print(f"   {it['arm']:28s} A={it['mean_a']:8.2f}(#{it['rank_a']:2d}) "
              f"B={it['mean_b']:8.1f}(#{it['rank_b']:2d})")
    return items


def main():
    all_rows = []

    # S1: MC framework vs baselines
    mc_fw = load(ROOT / "results" / "minecraft_search" / "framework_mc_detail.csv")
    mc_base = []
    for fn in ["baseline_mc_grid.csv", "baseline_mc_hill_climb.csv",
               "baseline_mc_random.csv"]:
        for r in load(ROOT / "results" / "minecraft_search" / fn):
            mc_base.append(dict(arm=fn.replace("baseline_mc_", "").replace(".csv", ""),
                                world_seed=r["world_seed"], algo_seed=r["algo_seed"],
                                points_found=r["points_found"],
                                total_queries=r["total_queries"]))
    fw_rows = [dict(arm="framework", world_seed=r["world_seed"],
                    algo_seed=r["algo_seed"], points_found=r["points_found"],
                    total_queries=r["total_queries"]) for r in mc_fw]
    s1 = summarize("S1_MC", "arm", fw_rows + mc_base, "points_found", "total_queries")
    all_rows += s1

    # S2: ablation_cross 三场景
    abl = load(ROOT / "results" / "ablation_cross" / "ablation_detail.csv")
    for sc in ["minecraft", "clouded_leopard", "chlorophyll"]:
        sub = [r for r in abl if r["scenario"] == sc]
        s2 = summarize(f"S2_{sc}", "variant", sub, "points_found", "total_queries")
        all_rows += s2

    # S3: 叶绿素 14 方法对比
    chl = load(ROOT / "results" / "chl_search" / "compare_oldnew" / "compare_detail.csv")
    s3 = summarize("S3_chl", "algo", chl, "points_found", "queries")
    all_rows += s3

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    print(f"[done] 输出 -> {OUT}")


if __name__ == "__main__":
    main()
