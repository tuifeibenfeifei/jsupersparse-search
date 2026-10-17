# -*- coding: utf-8 -*-
"""统一统计呈现：从已有 CSV 重算（不重跑实验）。

输入 ablation_detail.csv（3 场景 × full/AUC_off/multicluster_off）与 screen_mi_vs_auc.csv；
输出 peerj_stats_unified.csv（统一统计长表），另打印 Wilson CI 与锚点自检。
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


# 读 ablation
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
    """配对：variant - full；MC 按 (world,algo) 配对，其余按 algo_seed。"""
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
    """配对 Wilcoxon 符号秩（单尾）；全 0 diff 返回 p=1.0。"""
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

    # 锚点自检
    print("=== 锚点自检（chlorophyll 20-seed mean total_queries）===")
    for v, expect in [("full", 1596.25), ("AUC_off", 1805.0), ("multicluster_off", 958.0)]:
        vals = [r["total_queries"] for r in rows if r["scenario"] == "chlorophyll" and r["variant"] == v]
        m = float(np.mean(vals))
        ok = abs(m - expect) <= 5.0
        print(f"  chlorophyll/{v:16s} mean={m:8.2f}  expect~{expect:8.2f}  n={len(vals)}  {'OK' if ok else 'MISMATCH'}")

    # 统一表：每场景 × {AUC_off, multicluster_off} × {total_queries, points_found}
    # 单尾方向（与 harmful_arms_paired.csv 预注册假设一致）：
    #   AUC_off        : 关掉引导应使查询更多(greater)、命中更少(less)
    #   multicluster_off: 关掉多簇应使查询更少(less，H2)、命中更少(less)
    spec = {
        "AUC_off":          {"total_queries": "greater", "points_found": "less"},
        "multicluster_off":  {"total_queries": "less",    "points_found": "less"},
    }
    out = []
    print("\n=== 统一比较表（Δ = variant - full）===")
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
                    direction = "variant>full(off 更多)"
                elif dm < 0:
                    direction = "variant<full(off 更少)"
                else:
                    direction = "无差异"
                if np.isnan(p):
                    note = "无法检验(全同)"
                elif p < 0.05:
                    note = "p<0.05 显著"
                else:
                    note = "n.s.（p>=0.05，方向仅作假设）"
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
    print("\n=== 选择器 precision 0.763 的 Wilson 95% CI ===")
    scr_rows = []
    with open(SCR, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["sel_true"] == "":
                continue
            scr_rows.append(r)
    print(f"  有效逐事件行数(sel_true 非空) = {len(scr_rows)}")
    for screen in ["auc", "mi"]:
        sub = [r for r in scr_rows if r["screen"] == screen]
        tp = sum(int(r["sel_true"]) for r in sub)
        fp = sum(int(r["sel_noise"]) for r in sub)
        # 逐事件保留数 = sel_true+sel_noise（观测到 6 或 7 两种），pool 求和
        n = tp + fp
        phat = tp / n
        lo, hi = wilson_ci(phat, n)
        print(f"  [screen={screen:3s}] events={len(sub)} "
              f"TP(真信号保留)={tp}  FP(噪声保留)={fp}  n={n}  "
              f"p̂={phat:.4f}  Wilson95%CI=[{lo:.4f},{hi:.4f}]")
        print(f"        机会水平 7/19={7/19:.4f}；是否落在 CI 外: {not (lo <= 7/19 <= hi)}")


if __name__ == "__main__":
    main()
