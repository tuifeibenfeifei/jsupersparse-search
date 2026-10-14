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
"""screen_off_arms.py — 补跑 TABLE IX 缺失中间臂（AUC 屏幕关、引导保留）。

背景
----
手稿 TABLE IX（ISOLATED EFFECT OF AUC SCREENING, GUIDANCE KEPT ON；seeds 101–120, N=20）
为三臂设计：on（full）/ off（关 AUC 屏幕、保留引导）/ guidance off entirely。
其中中间臂 = 配置覆盖 {"sparse_guidance": False}（guidance_on 保持 True），
现有 results 中无当前版本数据源（ablation_cross 仅含 full / AUC_off[=引导整链关] / multicluster_off）。
本脚本用与 run_cross_ablation.py / heldout_seeds.py 完全相同的最终配置补跑该臂。

输出
----
  supplement_stats/screen_off_arms.csv          逐 seed 明细
  supplement_stats/screen_off_arms_summary.csv  统一配对检验（含既有三臂；单尾 Wilcoxon + Cliff's delta）
  supplement_stats/screen_off_arms.log          日志

用法
----
  $env:PYTHONPATH=REPO_ROOT / ".pylibs"
  python supplement_stats/screen_off_arms.py --canary   # 自检：full 配置 1 seed，与 ablation_cross 对照
  python supplement_stats/screen_off_arms.py --smoke 2  # 每场景 2 seed 试跑（不落盘）
  python supplement_stats/screen_off_arms.py            # 全量补跑 + 汇总
"""
import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(REPO_ROOT)
sys.path.insert(0, str(ROOT / "species-search"))
sys.path.insert(0, str(ROOT / "chlorophyll-search"))
sys.path.insert(0, str(ROOT))

from oracle import SpeciesOracle  # noqa: E402
from species_agent import SpeciesSearcher  # noqa: E402

HERE = ROOT / "supplement_stats"
ABL = ROOT / "results" / "ablation_cross"
DETAIL_OUT = HERE / "screen_off_arms.csv"
SUMMARY_OUT = HERE / "screen_off_arms_summary.csv"
LOG = HERE / "screen_off_arms.log"

SEEDS = list(range(101, 121))  # 与 TABLE IX 声明一致

# ---- 与 run_cross_ablation.py / heldout_seeds.py 完全相同的配置 ----
SPP_CFG = dict(
    target_points=20, max_queries=20000, min_dist_cells=1.0,
    stage2_min=3, stage3_min=5, stage3_jump_interval=30,
    stage3_jump_min_ratio=1.25, stage3_jump_max_ratio=5.0,
    curiosity_weight=1.0, learning_rate=0.1, discount_factor=0.95,
    epsilon_start=0.9, epsilon_end=0.1, epsilon_decay=0.999,
    stage1_grid_step=2, guidance_on=True, cov_in_state=True,
    recalc_every=20, stage2_topk=8,
    sparse_guidance=True, sparse_topk=6, sparse_min_pos=5,
    multicluster=True, cluster_ring_ratio=10.0, cluster_idle_limit=12,
    blank_jump=True, coverage_budget_ratio=0.12,
    blank_far_ratio=0.3, post_jump_grace=6, saturate_mask=False,
    cov_indices=list(range(19)), adaptive=True, directed_jump=True,
)

CHL_CFG = dict(
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

BASE_CFG = {"clouded_leopard": SPP_CFG, "chlorophyll": CHL_CFG}
ORACLE_PATHS = {
    "clouded_leopard": ROOT / "species-search" / "data" / "clouded_leopard" / "oracle.npz",
    "chlorophyll": ROOT / "chlorophyll-search" / "data" / "oracle.npz",
}
SCREEN_OFF_OVERRIDE = {"sparse_guidance": False}  # guidance_on 保持 True


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def wilcox_two_dir(a, b):
    """返回 (p_a_gt_b, p_a_lt_b)：单尾 Wilcoxon（a>b / a<b），与手稿口径一致（zero_method='wilcox'）。"""
    d = np.asarray(a, float) - np.asarray(b, float)
    p_gt = stats.wilcoxon(d, zero_method="wilcox", alternative="greater").pvalue
    p_lt = stats.wilcoxon(d, zero_method="wilcox", alternative="less").pvalue
    return float(p_gt), float(p_lt)


def cliff_delta(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("nan")
    sgn = sum(np.sum(b < x) - np.sum(b > x) for x in a)
    return sgn / (n * m)


def run_arm(scene, override, seeds):
    oracle = SpeciesOracle(ORACLE_PATHS[scene])
    cfg = dict(BASE_CFG[scene], **override)
    rows = []
    for ag in seeds:
        res = SpeciesSearcher(oracle, cfg).run_episode(ag)
        rows.append([ag, res["points_found"], res["total_queries"]])
    return rows


def load_abl(scene, variant):
    """从 ablation_cross/ablation_detail.csv 读某场景某臂，按 algo_seed 排序。"""
    out = {}
    with open(ABL / "ablation_detail.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["scenario"] == scene and r["variant"] == variant:
                out[int(r["algo_seed"])] = (float(r["points_found"]),
                                            float(r["total_queries"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canary", action="store_true", help="full 配置 1 seed 自检")
    ap.add_argument("--smoke", type=int, default=0, help="每场景 N seed 试跑")
    a = ap.parse_args()

    if LOG.exists():
        LOG.unlink()
    log("=== screen-off 臂补跑（TABLE IX 中间臂：sparse_guidance=False, guidance_on=True） ===")

    # ---------- canary ----------
    if a.canary:
        checks = [("chlorophyll", 101, 20, 1301), ("clouded_leopard", 101, 15, 20000)]
        all_ok = True
        for sc, seed, exp_p, exp_q in checks:
            t0 = time.time()
            rows = run_arm(sc, {}, [seed])
            ag, p, q = rows[0]
            ok = (p == exp_p and q == exp_q)
            all_ok &= ok
            log(f"canary {sc} seed={seed}: got pts={p} q={q} | expect pts={exp_p} q={exp_q} "
                f"-> {'MATCH' if ok else 'MISMATCH'} ({time.time()-t0:.0f}s)")
        log(f"canary result: {'ALL MATCH (环境与 ablation_cross 一致)' if all_ok else 'MISMATCH (需排查环境/线程设置)'}")
        return

    seeds = SEEDS[: a.smoke] if a.smoke else SEEDS

    # ---------- 1) 补跑 screen-off 臂 ----------
    t0 = time.time()
    all_rows = []
    for sc in ["chlorophyll", "clouded_leopard"]:
        rows = run_arm(sc, SCREEN_OFF_OVERRIDE, seeds)
        mp = sum(r[1] for r in rows) / len(rows)
        mq = sum(r[2] for r in rows) / len(rows)
        for ag, p, q in rows:
            all_rows.append([sc, "screen_off", 0, ag, p, q])
        log(f"  {sc:16s} screen_off pts={mp:.2f} q={mq:.1f} "
            f"({time.time()-t0:.0f}s elapsed, n={len(rows)})")

    if not a.smoke:
        with open(DETAIL_OUT, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["scenario", "variant", "world_seed", "algo_seed",
                        "points_found", "total_queries"])
            w.writerows(all_rows)
        log(f"[1/2] 明细 -> {DETAIL_OUT}")

    # ---------- 2) 统一配对检验（screen_off 新数据 + 既有三臂） ----------
    own = {sc: {"screen_off": {}} for sc in ["chlorophyll", "clouded_leopard"]}
    for sc, v, ws, ag, p, q in all_rows:
        own[sc][v][int(ag)] = (float(p), float(q))

    def block(sc, variant, metric):
        if variant == "screen_off":
            src = own[sc][variant]
        else:
            src = load_abl(sc, variant)
        if metric == "points":
            return [src[s][0] for s in sorted(src)]
        return [src[s][1] for s in sorted(src)]

    # (scene, tag, metric) — 全部以 a=full（全框架）、b=对照臂，与 harmful_arms_paired 数值口径一致
    comparisons = [
        ("chlorophyll", "full_vs_screen_off", "queries"),
        ("chlorophyll", "full_vs_screen_off", "points"),
        ("chlorophyll", "full_vs_guidance_off", "queries"),
        ("chlorophyll", "full_vs_multicluster_off", "queries"),
        ("clouded_leopard", "full_vs_screen_off", "points"),
        ("clouded_leopard", "full_vs_screen_off", "queries"),
        ("clouded_leopard", "full_vs_guidance_off", "points"),
        ("clouded_leopard", "full_vs_multicluster_off", "points"),
    ]
    summary = []
    log("\n--- 统一配对检验（n=%d；a=full, b=对照臂；p_gt 检验 a>b，p_lt 检验 a<b）---" % len(seeds))
    for sc, tag, metric in comparisons:
        other = tag.split("vs_", 1)[1]
        a_vals = block(sc, "full", metric)
        b_vals = block(sc, other, metric)
        n = min(len(a_vals), len(b_vals))
        a_vals, b_vals = a_vals[:n], b_vals[:n]
        ma, mb = float(np.mean(a_vals)), float(np.mean(b_vals))
        p_gt, p_lt = wilcox_two_dir(a_vals, b_vals)
        cd = cliff_delta(a_vals, b_vals)
        summary.append(dict(scene=sc, comparison=tag, metric=metric, n=n,
                            mean_a=round(ma, 3), mean_b=round(mb, 3),
                            diff=round(mb - ma, 3),
                            p_a_gt_b=round(p_gt, 4), p_a_lt_b=round(p_lt, 4),
                            cliff_delta=round(cd, 4)))
        log(f"  [{sc:16s}] {tag:28s} {metric:7s} "
            f"a={ma:9.2f} b={mb:9.2f} diff={mb-ma:+9.3f} "
            f"p(a>b)={p_gt:.4f} p(a<b)={p_lt:.4f} delta={cd:+.3f}")

    if not a.smoke:
        with open(SUMMARY_OUT, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        log(f"[2/2] 汇总 -> {SUMMARY_OUT}")
        log(f"=== 完成 ({(time.time()-t0)/60:.1f} min) ===")


if __name__ == "__main__":
    main()
