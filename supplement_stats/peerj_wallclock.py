# -*- coding: utf-8 -*-
"""peerj_wallclock.py — 端到端墙钟成本对照（PeerJ 论文第五章建议第 6 条）。

同预算同硬件下：
  场景 × 方法（framework / GP-UCB / TuRBO）
    - chlorophyll: 1 oracle x 20 algo seeds (101..120)，三方法全部实测计时
    - minecraft:   10 world seeds x 20 algo seeds（子集，见报告口径说明）；
                   framework 直接读既有 results/minecraft_search/framework_mc_detail.csv
                   （400 runs, wall_time_s 列）；GP-UCB/TuRBO 在同 npz oracle 上实测计时

代理增长微基准（chlorophyll 口径）：
  n in {100,500,1000,2000,4000}：
    GP 单步 = 重拟合 GPR(n 样本, 9 维) + 预测 2000 候选，重复 3 次取中位数
    框架单查询 = 1 个 CHL episode 的 wall_time / total_queries
  产出 supplement_stats/peerj_proxy_growth.csv

所有耗时用 time.perf_counter。所有数字来自实测或既有 CSV，不编造。
增量写盘：每完成一个 world 即追加 CSV，断点不丢。
"""
import csv
import math
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\19264\Desktop\deep-sea-exploration")
sys.path.insert(0, str(ROOT / "species-search"))
sys.path.insert(0, str(ROOT / "chlorophyll-search"))
sys.path.insert(0, str(ROOT / "chlorophyll-search" / "compare_oldnew"))
sys.path.insert(0, str(ROOT))

from oracle import SpeciesOracle  # noqa: E402
from species_agent import SpeciesSearcher  # noqa: E402
import run_compare as RC  # noqa: E402
from run_compare import Env, EpisodeCtx, algo_gp_ucb, algo_turbo, _fit_gp, _hist_features  # noqa: E402

OUT = ROOT / "supplement_stats"
OUT.mkdir(parents=True, exist_ok=True)
DETAIL_CSV = OUT / "peerj_wallclock_detail.csv"
PROXY_CSV = OUT / "peerj_proxy_growth.csv"
LOG = OUT / "peerj_wallclock.log"

SEEDS = list(range(101, 121))
MC_WORLD_SEEDS = list(range(1, 11))   # 子集：10 worlds（任务允许，报告标注）
RUN_WALL_WATCHDOG = 90.0               # 单 run 墙钟上限（秒），超过即中止并记录

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


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_rows(rows, mode="a"):
    """增量写明细 CSV。"""
    new = not DETAIL_CSV.exists() or mode == "w"
    with open(DETAIL_CSV, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "method", "world_seed", "algo_seed",
                                          "points_found", "total_queries", "wall_time_s"])
        if new:
            w.writeheader()
        w.writerows(rows)


# ===================== CHL 实测 =====================

def run_chl():
    rows = []
    oracle = SpeciesOracle(ROOT / "chlorophyll-search" / "data" / "oracle.npz")
    ctx = EpisodeCtx(oracle)

    for s in SEEDS:
        t0 = time.perf_counter()
        res = SpeciesSearcher(oracle, CHL_CFG).run_episode(s)
        dt = time.perf_counter() - t0
        rows.append(dict(scenario="chlorophyll", method="framework", world_seed=0,
                         algo_seed=s, points_found=res["points_found"],
                         total_queries=res["total_queries"], wall_time_s=round(dt, 4)))
        log(f"  chl/framework seed={s} pts={res['points_found']} q={res['total_queries']} wall={dt:.3f}s")

    for s in SEEDS:
        env = Env(oracle)
        rng = random.Random(s)
        t0 = time.perf_counter()
        algo_gp_ucb(env, rng, ctx)
        dt = time.perf_counter() - t0
        r = env.result()
        rows.append(dict(scenario="chlorophyll", method="gp_ucb", world_seed=0,
                         algo_seed=s, points_found=r["points_found"],
                         total_queries=r["total_queries"], wall_time_s=round(dt, 4)))
        log(f"  chl/gp_ucb seed={s} pts={r['points_found']} q={r['total_queries']} wall={dt:.3f}s")

    for s in SEEDS:
        env = Env(oracle)
        rng = random.Random(s)
        t0 = time.perf_counter()
        algo_turbo(env, rng, ctx)
        dt = time.perf_counter() - t0
        r = env.result()
        rows.append(dict(scenario="chlorophyll", method="turbo", world_seed=0,
                         algo_seed=s, points_found=r["points_found"],
                         total_queries=r["total_queries"], wall_time_s=round(dt, 4)))
        log(f"  chl/turbo seed={s} pts={r['points_found']} q={r['total_queries']} wall={dt:.3f}s")
    return rows


# ===================== MC =====================

class MCSimpleCtx:
    def __init__(self, oracle: SpeciesOracle):
        H, W = oracle.nrows, oracle.ncols
        ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        coord = np.stack([ii / (H - 1), jj / (W - 1)], axis=-1).astype(np.float32)
        covf = np.zeros((H, W, oracle.covs.shape[0]), dtype=np.float32)
        for k in range(oracle.covs.shape[0]):
            v = oracle.covs[k].astype(np.float32)
            lo, hi = float(v.min()), float(v.max())
            if hi - lo < 1e-9:
                hi = lo + 1.0
            covf[..., k] = (v - lo) / (hi - lo)
        self.Xgrid = np.concatenate([coord, covf], axis=-1)
        self.H, self.W = H, W


def _mc_gp_ucb(env, rng, ctx, cold=12, kappa=2.0, pool=2000, batch=20,
               hist_n=300, t_start=None):
    H, W = ctx.H, ctx.W
    while not env.done and env.q < cold:
        env.query(rng.randrange(H), rng.randrange(W))
    queued = []
    while not env.done:
        if t_start is not None and time.perf_counter() - t_start > RUN_WALL_WATCHDOG:
            log(f"    [watchdog] gp_ucb ws={env.o.world_seed if hasattr(env.o,'world_seed') else '?'} q={env.q} aborted at {RUN_WALL_WATCHDOG}s")
            break
        if not queued:
            Xtr, ytr = _hist_features(ctx, env, n=hist_n)
            g = _fit_gp(Xtr, ytr)
            if g is None:
                queued = [(rng.randrange(H), rng.randrange(W)) for _ in range(batch)]
            else:
                cand, Xc = RC._gp_candidates(ctx, env, rng, pool=pool)
                if not cand:
                    break
                mu, std = g.predict(Xc, return_std=True)
                acq = mu + kappa * std
                order = np.argsort(acq)[::-1][:batch]
                queued = [cand[k] for k in order]
        i, j = queued.pop(0)
        env.query(i, j)


def _mc_turbo(env, rng, ctx, cold=12, pool=300, batch=8, init_radius=0.25,
              hist_n=250, t_start=None):
    H, W = ctx.H, ctx.W
    while not env.done and env.q < cold:
        env.query(rng.randrange(H), rng.randrange(W))
    center = np.array([rng.uniform(0, H - 1), rng.uniform(0, W - 1)])
    radius = init_radius * max(H, W)
    queued = []
    while not env.done:
        if t_start is not None and time.perf_counter() - t_start > RUN_WALL_WATCHDOG:
            log(f"    [watchdog] turbo q={env.q} aborted at {RUN_WALL_WATCHDOG}s")
            break
        if not queued:
            Xtr, ytr = _hist_features(ctx, env, n=hist_n)
            g = _fit_gp(Xtr, ytr)
            n_samp = pool * 3
            offs = np.array([rng.gauss(0, radius) for _ in range(n_samp * 2)]).reshape(n_samp, 2)
            pts = center[None, :] + offs
            cands = []
            for pt in pts:
                i = int(round(pt[0])); j = int(round(pt[1]))
                if 0 <= i < H and 0 <= j < W and (i, j) not in env.visited:
                    cands.append((i, j))
                    if len(cands) >= pool:
                        break
            if len(cands) < batch:
                for _ in range(batch - len(cands)):
                    i = rng.randrange(H); j = rng.randrange(W)
                    if (i, j) not in env.visited:
                        cands.append((i, j))
            if g is None or not cands:
                queued = cands[:batch]
            else:
                Xc = np.array([ctx.Xgrid[i, j] for (i, j) in cands])
                mu, std = g.predict(Xc, return_std=True)
                acq = mu + 1.5 * std
                order = np.argsort(acq)[::-1][:batch]
                queued = [cands[k] for k in order]
            if not queued:
                break
        i, j = queued.pop(0)
        pres, cov, is_new = env.query(i, j)
        if pres is None:
            break
        if is_new:
            center = np.array([float(i), float(j)])
            radius = min(radius * 1.5, 0.5 * max(H, W))
        else:
            radius *= 0.98
        if radius < 2.0:
            center = np.array([rng.uniform(0, H - 1), rng.uniform(0, W - 1)])
            radius = init_radius * max(H, W)


def run_mc_framework_existing():
    p = ROOT / "results" / "minecraft_search" / "framework_mc_detail.csv"
    rows = []
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(dict(scenario="minecraft", method="framework",
                             world_seed=int(r["world_seed"]), algo_seed=int(r["algo_seed"]),
                             points_found=int(r["points_found"]),
                             total_queries=int(r["total_queries"]),
                             wall_time_s=float(r["wall_time_s"])))
    log(f"  mc/framework 读既有 CSV {len(rows)} runs")
    return rows


def run_mc_agents():
    # Env 口径切到 MC
    RC.TARGET = 40
    RC.MAX_Q = 10000
    RC.MIN_D = 2.0
    for ws in MC_WORLD_SEEDS:
        oracle = SpeciesOracle(ROOT / "minecraft-search" / "oracle_npz" / f"mc_world_seed{ws}.npz")
        ctx = MCSimpleCtx(oracle)
        world_rows = []
        for s in SEEDS:
            # gp_ucb
            env = Env(oracle)
            rng = random.Random(s)
            t0 = time.perf_counter()
            _mc_gp_ucb(env, rng, ctx, t_start=t0)
            dt = time.perf_counter() - t0
            r = env.result()
            world_rows.append(dict(scenario="minecraft", method="gp_ucb", world_seed=ws,
                                   algo_seed=s, points_found=r["points_found"],
                                   total_queries=r["total_queries"], wall_time_s=round(dt, 4)))
            # turbo
            env = Env(oracle)
            rng = random.Random(s + 1000)
            t0 = time.perf_counter()
            _mc_turbo(env, rng, ctx, t_start=t0)
            dt = time.perf_counter() - t0
            r = env.result()
            world_rows.append(dict(scenario="minecraft", method="turbo", world_seed=ws,
                                   algo_seed=s, points_found=r["points_found"],
                                   total_queries=r["total_queries"], wall_time_s=round(dt, 4)))
        write_rows(world_rows)  # 增量落盘
        log(f"  mc agents world_seed={ws} done (+{len(world_rows)} rows)")


# ===================== 代理增长微基准 =====================

def run_proxy_growth():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
    rng = np.random.RandomState(0)
    rows = []
    oracle = SpeciesOracle(ROOT / "chlorophyll-search" / "data" / "oracle.npz")
    ctx = EpisodeCtx(oracle)
    Xgrid = ctx.Xgrid.reshape(-1, ctx.Xgrid.shape[1])

    for n in [100, 500, 1000, 2000, 4000]:
        idx = rng.choice(len(Xgrid), size=n, replace=False)
        Xtr = Xgrid[idx].astype(np.float64)
        ytr = rng.RandomState(n).randn(n) * 0.1
        cand_idx = rng.choice(len(Xgrid), size=2000, replace=False)
        Xcand = Xgrid[cand_idx].astype(np.float64)
        times = []
        for rep in range(3):
            t0 = time.perf_counter()
            g = GaussianProcessRegressor(kernel=RBF(length_scale=0.5), optimizer=None,
                                         alpha=1e-3, normalize_y=False)
            g.fit(Xtr, ytr)
            _ = g.predict(Xcand, return_std=True)
            times.append(time.perf_counter() - t0)
        med = statistics.median(times)
        rows.append(dict(n=n, gp_fit_plus_predict_s=round(med, 5),
                         reps=",".join(f"{t:.5f}" for t in times)))
        log(f"  proxy n={n:5d} gp_step_median={med*1000:.2f} ms")

    oracle2 = SpeciesOracle(ROOT / "chlorophyll-search" / "data" / "oracle.npz")
    t0 = time.perf_counter()
    res = SpeciesSearcher(oracle2, CHL_CFG).run_episode(101)
    fw_wall = time.perf_counter() - t0
    fw_per_q = fw_wall / max(res["total_queries"], 1)
    log(f"  framework 1 episode: wall={fw_wall:.3f}s q={res['total_queries']} per_q={fw_per_q*1000:.3f} ms")
    for r in rows:
        r["framework_per_query_s"] = round(fw_per_q, 6)
        r["ratio_gp_over_framework"] = round(r["gp_fit_plus_predict_s"] / fw_per_q, 1)

    with open(PROXY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["n", "gp_fit_plus_predict_s", "reps",
                                         "framework_per_query_s", "ratio_gp_over_framework"])
        w.writeheader()
        w.writerows(rows)
    log(f"  proxy growth -> {PROXY_CSV}")
    return rows


# ===================== 主流程 =====================

def main():
    t_all = time.perf_counter()
    if LOG.exists():
        LOG.unlink()
    if DETAIL_CSV.exists():
        DETAIL_CSV.unlink()
    log("=== PeerJ 端到端墙钟成本对照 ===")

    log("[1/4] chlorophyll 三方法实测 20 seeds ...")
    chl_rows = run_chl()
    write_rows(chl_rows, mode="w")

    log("[2/4] minecraft framework 读既有 CSV ...")
    fw_rows = run_mc_framework_existing()
    write_rows(fw_rows)

    log("[3/4] minecraft GP-UCB/TuRBO 实测 10 worlds x 20 seeds ...")
    run_mc_agents()

    log("[4/4] 代理增长微基准 ...")
    run_proxy_growth()

    dt = time.perf_counter() - t_all
    log(f"=== 全部完成 总耗时 {dt/60:.1f} min ===")


if __name__ == "__main__":
    main()
