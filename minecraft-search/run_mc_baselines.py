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
"""run_mc_baselines.py — Minecraft 游戏场景：random/grid/hill_climb 三基线对比。

口径（与框架 final_model.yaml 对齐）：
  - 世界种子 1-20（与框架 run_final_model.py 默认 seeds 1-20 完全一致）。
  - 每世界种子 × 每方法跑 5 次（algo_seed=101..105），共 20×5=100 次/方法。
  - 目标 target=40 深海格，预算 max_queries=10000，min_dist_cells=2.0。
  - grid_step 取 10（100x100 网格每 10 格扫一次，与框架 stage1 盲扫密度同量级）。

输出 results/minecraft_search/：
  baseline_mc_<method>.csv   逐 run 明细
  mc_baseline_summary.csv   方法汇总
"""
import csv
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(REPO_ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "minecraft-search"))

from mc_oracle import MCGridOracle  # noqa: E402

OUT = ROOT / "results" / "minecraft_search"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "progress.log"

WORLD_SEEDS = list(range(1, 21))          # 与框架对齐 1-20
ALGO_SEEDS = list(range(101, 106))         # 每世界种子跑 5 次
TARGET = 40
MAX_Q = 10000
MIN_D = 2.0
GRID_STEP = 10


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------- baselines ----------------
_D8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
       (1, -1), (1, 0), (1, 1)]


def _mindist_ok(i, j, found, min_d):
    return all(math.hypot(i - fi, j - fj) >= min_d for fi, fj in found)


def random_baseline(oracle, algo_seed):
    rng = random.Random(algo_seed)
    nrows, ncols = oracle.nrows, oracle.ncols
    found = []
    oracle.reset_counter()
    while oracle.query_count < MAX_Q and len(found) < TARGET:
        i, j = rng.randrange(nrows), rng.randrange(ncols)
        p, _ = oracle.query_cell(i, j)
        if p and _mindist_ok(i, j, found, MIN_D):
            found.append((i, j))
    return len(found), oracle.query_count


def grid_baseline(oracle, algo_seed):
    rng = random.Random(algo_seed)
    nrows, ncols = oracle.nrows, oracle.ncols
    # 随机偏移起扫的规则网格
    off_i, off_j = rng.randrange(GRID_STEP), rng.randrange(GRID_STEP)
    found = []
    oracle.reset_counter()
    i = off_i
    while i < nrows and oracle.query_count < MAX_Q and len(found) < TARGET:
        j = off_j
        while j < ncols and oracle.query_count < MAX_Q and len(found) < TARGET:
            p, _ = oracle.query_cell(i, j)
            if p and _mindist_ok(i, j, found, MIN_D):
                found.append((i, j))
            j += GRID_STEP
        i += GRID_STEP
    return len(found), oracle.query_count


def hill_climb(oracle, algo_seed):
    rng = random.Random(algo_seed)
    nrows, ncols = oracle.nrows, oracle.ncols
    stuck_limit = 6
    found = []
    oracle.reset_counter()
    cur = (rng.randrange(nrows), rng.randrange(ncols))
    climbed = 0
    while oracle.query_count < MAX_Q and len(found) < TARGET:
        i, j = cur
        p, _ = oracle.query_cell(i, j)
        if p is None:
            cur = (rng.randrange(nrows), rng.randrange(ncols))
            continue
        if p:
            if _mindist_ok(i, j, found, MIN_D):
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
                    if _mindist_ok(ni, nj, found, MIN_D):
                        found.append((ni, nj))
                    cur = (ni, nj)
                    moved = True
                    break
            if not moved:
                climbed += 1
                if climbed >= stuck_limit:
                    cur = (rng.randrange(nrows), rng.randrange(ncols))
                    climbed = 0
                else:
                    d = rng.choice(_D8)
                    cur = (min(nrows - 1, max(0, i + d[0])),
                           min(ncols - 1, max(0, j + d[1])))
        else:
            cur = (rng.randrange(nrows), rng.randrange(ncols))
    return len(found), oracle.query_count


METHODS = {"random": random_baseline, "grid": grid_baseline, "hill_climb": hill_climb}


def main():
    t_all = time.time()
    summary = []
    log(f"==== Minecraft 深海(海底神殿)基线对比: world_seeds={WORLD_SEEDS[0]}-{WORLD_SEEDS[-1]} "
        f"algo_seeds={ALGO_SEEDS} target={TARGET} budget={MAX_Q} grid_step={GRID_STEP}")

    for mkey, mfn in METHODS.items():
        rows = []
        t0 = time.time()
        for ws in WORLD_SEEDS:
            oracle = MCGridOracle(world_seed=ws)
            for ag in ALGO_SEEDS:
                pts, qs = mfn(oracle, ag)
                rows.append([ws, ag, pts, qs])
        det = OUT / f"baseline_mc_{mkey}.csv"
        with open(det, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["world_seed", "algo_seed", "points_found", "total_queries"])
            w.writerows(rows)
        mp = sum(r[2] for r in rows) / len(rows)
        mq = sum(r[3] for r in rows) / len(rows)
        hit = sum(1 for r in rows if r[2] >= TARGET) / len(rows)
        dt = time.time() - t0
        log(f"  {mkey:11s} pts={mp:.2f}/{TARGET} q={mq:.0f} 达标率={hit*100:.0f}% ({dt:.1f}s)")
        summary.append([mkey, round(mp, 2), round(mq, 1), round(hit, 3),
                        min(r[2] for r in rows), max(r[2] for r in rows)])

    with open(OUT / "mc_baseline_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "mean_points", "mean_queries", "hit_rate", "min_points", "max_points"])
        w.writerows(summary)
    log(f"==== 完成 ({(time.time()-t_all)/60:.1f}min) -> {OUT}")


if __name__ == "__main__":
    main()
