# -*- coding: utf-8 -*-
"""续跑：MC worlds 7-10，追加到既有 detail CSV。"""
import csv, random, sys, time
from datetime import datetime
from pathlib import Path
ROOT = Path(r"C:\Users\19264\Desktop\deep-sea-exploration")
sys.path.insert(0, str(ROOT / "species-search"))
sys.path.insert(0, str(ROOT / "chlorophyll-search"))
sys.path.insert(0, str(ROOT / "chlorophyll-search" / "compare_oldnew"))
sys.path.insert(0, str(ROOT))
import numpy as np
from oracle import SpeciesOracle  # noqa: E402
import run_compare as RC  # noqa: E402
from run_compare import Env, _fit_gp, _hist_features  # noqa: E402

OUT = ROOT / "supplement_stats"
DETAIL_CSV = OUT / "peerj_wallclock_detail.csv"
LOG = OUT / "peerj_wallclock.log"
SEEDS = list(range(101, 121))
WATCH = 20.0

def log(m):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")

class MCSimpleCtx:
    def __init__(self, oracle):
        H, W = oracle.nrows, oracle.ncols
        ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        coord = np.stack([ii / (H - 1), jj / (W - 1)], axis=-1).astype(np.float32)
        covf = np.zeros((H, W, oracle.covs.shape[0]), dtype=np.float32)
        for k in range(oracle.covs.shape[0]):
            v = oracle.covs[k].astype(np.float32)
            lo, hi = float(v.min()), float(v.max())
            if hi - lo < 1e-9: hi = lo + 1.0
            covf[..., k] = (v - lo) / (hi - lo)
        self.Xgrid = np.concatenate([coord, covf], axis=-1)
        self.H, self.W = H, W

def gp_ucb(env, rng, ctx, cold=12, kappa=2.0, pool=2000, batch=20, t0=None):
    H, W = ctx.H, ctx.W
    while not env.done and env.q < cold:
        env.query(rng.randrange(H), rng.randrange(W))
    queued = []
    while not env.done:
        if t0 and time.perf_counter() - t0 > WATCH:
            log(f"    [watchdog gp_ucb] q={env.q}"); break
        if not queued:
            Xtr, ytr = _hist_features(ctx, env, n=300)
            g = _fit_gp(Xtr, ytr)
            if g is None:
                queued = [(rng.randrange(H), rng.randrange(W)) for _ in range(batch)]
            else:
                cand, Xc = RC._gp_candidates(ctx, env, rng, pool=pool)
                if not cand: break
                mu, std = g.predict(Xc, return_std=True)
                order = np.argsort(mu + kappa * std)[::-1][:batch]
                queued = [cand[k] for k in order]
        i, j = queued.pop(0)
        env.query(i, j)

def turbo(env, rng, ctx, cold=12, pool=300, batch=8, init_radius=0.25, t0=None):
    H, W = ctx.H, ctx.W
    while not env.done and env.q < cold:
        env.query(rng.randrange(H), rng.randrange(W))
    center = np.array([rng.uniform(0, H - 1), rng.uniform(0, W - 1)])
    radius = init_radius * max(H, W)
    queued = []
    while not env.done:
        if t0 and time.perf_counter() - t0 > WATCH:
            log(f"    [watchdog turbo] q={env.q}"); break
        if not queued:
            Xtr, ytr = _hist_features(ctx, env, n=250)
            g = _fit_gp(Xtr, ytr)
            n_samp = pool * 3
            offs = np.array([rng.gauss(0, radius) for _ in range(n_samp * 2)]).reshape(n_samp, 2)
            pts = center[None, :] + offs
            cands = []
            for pt in pts:
                i = int(round(pt[0])); j = int(round(pt[1]))
                if 0 <= i < H and 0 <= j < W and (i, j) not in env.visited:
                    cands.append((i, j))
                    if len(cands) >= pool: break
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
                order = np.argsort(mu + 1.5 * std)[::-1][:batch]
                queued = [cands[k] for k in order]
            if not queued: break
        i, j = queued.pop(0)
        pres, cov, is_new = env.query(i, j)
        if pres is None: break
        if is_new:
            center = np.array([float(i), float(j)])
            radius = min(radius * 1.5, 0.5 * max(H, W))
        else:
            radius *= 0.98
        if radius < 2.0:
            center = np.array([rng.uniform(0, H - 1), rng.uniform(0, W - 1)])
            radius = init_radius * max(H, W)

def main():
    RC.TARGET = 40; RC.MAX_Q = 10000; RC.MIN_D = 2.0
    # 已完成的 worlds
    done = set()
    with open(DETAIL_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["scenario"] == "minecraft" and r["method"] in ("gp_ucb", "turbo"):
                done.add(int(r["world_seed"]))
    log(f"[resume] 已完成 MC agent worlds: {sorted(done)}")
    for ws in range(7, 11):
        if ws in done:
            continue
        oracle = SpeciesOracle(ROOT / "minecraft-search" / "oracle_npz" / f"mc_world_seed{ws}.npz")
        ctx = MCSimpleCtx(oracle)
        rows = []
        for s in SEEDS:
            env = Env(oracle); rng = random.Random(s); t0 = time.perf_counter()
            gp_ucb(env, rng, ctx, t0=t0); dt = time.perf_counter() - t0; r = env.result()
            rows.append(dict(scenario="minecraft", method="gp_ucb", world_seed=ws, algo_seed=s,
                              points_found=r["points_found"], total_queries=r["total_queries"],
                              wall_time_s=round(dt, 4)))
            env = Env(oracle); rng = random.Random(s + 1000); t0 = time.perf_counter()
            turbo(env, rng, ctx, t0=t0); dt = time.perf_counter() - t0; r = env.result()
            rows.append(dict(scenario="minecraft", method="turbo", world_seed=ws, algo_seed=s,
                              points_found=r["points_found"], total_queries=r["total_queries"],
                              wall_time_s=round(dt, 4)))
        with open(DETAIL_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["scenario","method","world_seed","algo_seed",
                                              "points_found","total_queries","wall_time_s"])
            w.writerows(rows)
        log(f"  mc agents world_seed={ws} done (+{len(rows)} rows)")
    log("[resume] 完成")

if __name__ == "__main__":
    main()
