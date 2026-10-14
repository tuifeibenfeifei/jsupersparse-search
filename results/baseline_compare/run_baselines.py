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
"""run_baselines.py — 任务1：4 新场景加 random / grid / hill_climb 三基线对比。

口径（务必读完）：
  - 单阶段场景（污染 pollution、海上搜救 maritime）：基线直接在主 oracle 上跑，
    target / max_queries / min_dist 与框架 regress_v2 完全同款。
  - 两阶段场景（逆向漂移 drift、矿化 mineral）：采用「只比阶段 A」口径（任务书
    方案 i）。基线在阶段 A oracle 上跑（同 target/budget/min_dist），与框架
    regress_v2 的 A_points / A_queries 直接对比。不跑 B：B 阶段依赖 A 的解码
    （漂移：质心回推 + 追加距离波段；矿化：RANSAC 解产状 + 3D Searcher3D 注入
    预测面），纯黑盒 random/grid/hill_climb 拿不到这些引导协变量，硬跑 B 既不
    公平也不成比例；A 阶段搜索能力才是四种方法 apples-to-apples 的对象。
  - grid 基线的 grid_step 取框架该场景 stage1_grid_step（相同盲扫空间密度）。
  - seeds 101-110 ×10，与框架同种子集。单进程串行。

输出：
  results/baseline_compare/
    baseline_<场景>_<方法>.csv   逐 seed 明细
    baseline_summary.csv        场景×方法 汇总（含框架列）
    （图与报告由 finalize_baselines.py 生成）
"""
import csv
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(REPO_ROOT)
for p in ("species-search", "pollution-search", "maritime-search",
          "drift-search", "mineral-search"):
    sys.path.insert(0, str(ROOT / p))

from species_agent import random_baseline, grid_baseline   # noqa: E402
from pollution_oracle import PollutionOracle                # noqa: E402
from build_maritime_oracle import MaritimeOracle            # noqa: E402
from drift_oracle import DriftOracle                        # noqa: E402
from build_mineral_oracle import MineralOracle2D            # noqa: E402

OUT = ROOT / "results" / "baseline_compare"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "progress.log"
SEEDS = list(range(101, 111))


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ==================== 黑盒爬山（自包含，复用 tv_baselines.hillclimb_episode 语义）====================
_D8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
       (1, -1), (1, 0), (1, 1)]


def hill_climb(oracle, config: dict, algo_seed: int) -> dict:
    """随机起点 → 查询当前格 → 沿 8 邻域贪心爬 presence；连续 stuck 步无推进则随机重启。
    黑盒：只调用 oracle.query_cell。命中邻域即逐格查询（每查询都计预算/时间步）。"""
    rng = random.Random(algo_seed)
    target = int(config.get("target_points", 20))
    max_q = int(config.get("max_queries", 1500))
    min_d = float(config.get("min_dist_cells", 2.0))
    stuck_limit = int(config.get("hc_stuck_limit", 6))
    nrows, ncols = oracle.nrows, oracle.ncols

    def rand_cell():
        return rng.randrange(nrows), rng.randrange(ncols)

    def mindist_ok(i, j, found):
        return all(math.hypot(i - fi, j - fj) >= min_d for fi, fj in found)

    found = []
    oracle.reset_counter()
    cur = rand_cell()
    climbed = 0
    while oracle.query_count < max_q and len(found) < target:
        i, j = cur
        pres, _ = oracle.query_cell(i, j)
        if pres is None:
            cur = rand_cell()
            continue
        if pres:
            if mindist_ok(i, j, found):
                found.append((i, j))
            order = list(_D8)
            rng.shuffle(order)
            moved = False
            for di, dj in order:
                ni, nj = i + di, j + dj
                if not (0 <= ni < nrows and 0 <= nj < ncols):
                    continue
                p2, _ = oracle.query_cell(ni, nj)
                if p2:
                    if mindist_ok(ni, nj, found):
                        found.append((ni, nj))
                    cur = (ni, nj)
                    moved = True
                    break
            if not moved:
                climbed += 1
                if climbed >= stuck_limit:
                    cur = rand_cell()
                    climbed = 0
                else:
                    d = rng.choice(_D8)
                    cur = (min(nrows - 1, max(0, i + d[0])),
                           min(ncols - 1, max(0, j + d[1])))
        else:
            cur = rand_cell()
    return {"points_found": len(found), "total_queries": oracle.query_count,
            "found_cells": found}


METHODS = {
    "random": random_baseline,
    "grid": grid_baseline,
    "hill_climb": hill_climb,
}


def main():
    # (场景key, 显示名, oracle工厂, cfg, 目标点数)
    scenarios = [
        ("pollution", "污染溯源",
         lambda: PollutionOracle(),
         dict(target_points=15, max_queries=800, min_dist_cells=1.0, grid_step=3),
         15),
        ("maritime", "海上搜救",
         lambda: MaritimeOracle(ROOT / "maritime-search" / "data" / "oracle.npz"),
         dict(target_points=15, max_queries=600, min_dist_cells=1.0, grid_step=4),
         15),
        ("driftA", "逆向漂移(阶段A)",
         lambda: DriftOracle(ROOT / "drift-search" / "data" / "stageA.npz"),
         dict(target_points=6, max_queries=400, min_dist_cells=2.0, grid_step=8),
         6),
        ("mineralA", "矿化露头(阶段A)",
         lambda: MineralOracle2D(ROOT / "mineral-search" / "data" / "stageA_surface.npz"),
         dict(target_points=10, max_queries=500, min_dist_cells=1.0, grid_step=8),
         10),
    ]

    summary_rows = []
    t_all = time.time()
    for skey, sname, oracle_fn, cfg, target in scenarios:
        oracle = oracle_fn()
        log(f"==== 场景 {sname} ({skey}): oracle {oracle.nrows}x{oracle.ncols} "
            f"target={target} budget={cfg['max_queries']} min_dist={cfg['min_dist_cells']} "
            f"grid_step={cfg['grid_step']} seeds={SEEDS}")
        for mkey, mfn in METHODS.items():
            det_path = OUT / f"baseline_{skey}_{mkey}.csv"
            rows = []
            t0 = time.time()
            for seed in SEEDS:
                res = mfn(oracle, cfg, seed)
                rows.append([seed, res["points_found"], res["total_queries"]])
            with open(det_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["seed", "points_found", "total_queries"])
                w.writerows(rows)
            mp = sum(r[1] for r in rows) / len(rows)
            mq = sum(r[2] for r in rows) / len(rows)
            hit_rate = sum(1 for r in rows if r[1] >= target) / len(rows)
            dt = time.time() - t0
            log(f"  {mkey:11s} pts={mp:.2f}/{target} q={mq:.0f} "
                f"达标率={hit_rate*100:.0f}% ({dt:.1f}s)")
            summary_rows.append([skey, sname, target, mkey,
                                 round(mp, 3), round(mq, 1),
                                 round(hit_rate, 3),
                                 min(r[1] for r in rows), max(r[1] for r in rows)])

    with open(OUT / "baseline_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario_key", "scenario", "target", "method",
                    "mean_points_found", "mean_total_queries", "hit_rate",
                    "min_points", "max_points"])
        w.writerows(summary_rows)
    log(f"==== 全部基线完成 ({(time.time()-t_all)/60:.1f}min) -> {OUT}")


if __name__ == "__main__":
    main()
