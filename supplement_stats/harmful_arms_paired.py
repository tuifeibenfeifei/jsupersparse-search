# -*- coding: utf-8 -*-
import os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve()
for _ in range(8):
    if (REPO_ROOT / 'configs').is_dir():
        break
    REPO_ROOT = REPO_ROOT.parent
"""TABLE XV paired test confirmation of four "harmful arms".

H1 guidance @ Clouded leopard(full vs AUC_off),H2 multicluster @ Chlorophyll(full vs multicluster_off),
H3 multicluster @ 3-D hypoxic (O3D5 vs O3D6/O3D7), H4 framework vs greedy/hill_climb @ Chlorophyll, 20 pairs each.
Statistics consistent with manuscript TABLE XI: one-tailed Wilcoxon signed-rank test + Cliff's delta.
Outputs supplement_stats/harmful_arms_paired.csv + console summary.
"""
import csv
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[0]
ROOT = Path(REPO_ROOT)
sys.path.insert(0, str(ROOT / "chlorophyll-search"))
from oracle import SpeciesOracle  # noqa: E402

OUT = HERE / "harmful_arms_paired.csv"
LOG = HERE / "harmful_arms_paired.log"

SEEDS = list(range(101, 121))   # 20 seeds (consistent with ablation_cross)
TARGET = 20
MAX_Q = 4000
MIN_D = 1.0
STRUCT = list(range(7))


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_csv(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def wilcox_onesided(a, b, direction):
    """direction: 'a_better' tests whether a is significantly better than b (a-b>0 one-tailed);
    'b_better' tests whether b is significantly better than a (a-b<0 one-tailed). Returns p value."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if direction == "a_better":
        return stats.wilcoxon(d, zero_method="wilcox", alternative="greater").pvalue
    else:
        return stats.wilcoxon(d, zero_method="wilcox", alternative="less").pvalue


def cliff_delta(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = len(a); m = len(b)
    if n == 0 or m == 0:
        return float("nan")
    sgn = 0
    for x in a:
        sgn += np.sum(b < x) - np.sum(b > x)
    return sgn / (n * m)


def paired_stats(tag, scene, metric, a, b, direction, claim):
    """a = control/off/baseline arm; b = full framework arm. direction specifies the significance direction supporting the "harmful" claim."""
    a = [float(x) for x in a]; b = [float(x) for x in b]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ma, mb = float(np.mean(a)), float(np.mean(b))
    p = wilcox_onesided(a, b, direction)
    cd = cliff_delta(a, b)
    row = dict(arm=tag, scene=scene, metric=metric, n=n,
               mean_a=round(ma, 3), mean_b=round(mb, 3),
               diff=round(mb - ma, 3),
               wilcox_p=round(p, 4), cliff_delta=round(cd, 4), claim=claim)
    sig = "sig" if p < 0.05 else ("marginal" if p < 0.1 else "ns")
    log(f"[{tag}] {metric:8s} {scene:16s} a={ma:8.2f} b={mb:8.2f} "
        f"diff={mb - ma:+8.2f} p={p:.4f}({sig}) delta={cd:+.3f} {claim}")
    return row


# Chlorophyll baselines (rerun for H4)

class Env:
    def __init__(self, oracle):
        self.o = oracle
        self.reset()

    def reset(self):
        self.o.reset_counter()
        self.found = []
        self.visited = set()
        self.pos_cov = []
        self.X_hist = []
        self.y_hist = []

    @property
    def q(self):
        return self.o.query_count

    @property
    def done(self):
        return self.q >= MAX_Q or len(self.found) >= TARGET

    def query(self, i, j):
        i = int(i); j = int(j)
        pres, cov = self.o.query_cell(i, j)
        if pres is None:
            return None
        self.visited.add((i, j))
        is_new = False
        if pres:
            if all(math.hypot(i - fi, j - fj) >= MIN_D for fi, fj in self.found):
                self.found.append((i, j))
                self.pos_cov.append(cov.copy())
                is_new = True
        self.X_hist.append((i, j))
        self.y_hist.append(1.0 if pres else 0.0)
        return pres, cov, is_new

    def result(self):
        return dict(points_found=len(self.found), total_queries=self.q,
                    covered_cells=len(self.visited))


def build_feature_fields(o):
    H, W = o.nrows, o.ncols
    covf = np.zeros((H, W, len(STRUCT)), dtype=np.float32)
    for kk, k in enumerate(STRUCT):
        v = o.covs[k].astype(np.float32)
        lo, hi = float(v.min()), float(v.max())
        if hi - lo < 1e-9:
            hi = lo + 1.0
        covf[..., kk] = (v - lo) / (hi - lo)
    return covf


class EpisodeCtx:
    def __init__(self, o):
        self.Xgrid = build_feature_fields(o)
        self.H, self.W = o.nrows, o.ncols
        mins, maxs = [], []
        for k in STRUCT:
            v = o.covs[k].astype(np.float32)
            mins.append(float(v.min())); maxs.append(float(v.max()))
        self._cov_min = np.array(mins, dtype=np.float32)
        self._cov_max = np.array(maxs, dtype=np.float32)

    def norm_pos_cov(self, pos_cov_raw_list):
        if not pos_cov_raw_list:
            return np.zeros((0, 7), dtype=np.float32)
        arr = np.array(pos_cov_raw_list)[:, STRUCT]
        arr = (arr - self._cov_min) / (self._cov_max - self._cov_min + 1e-9)
        return arr.astype(np.float32)


def proxy_score_map(Xgrid, pos_cov_norm):
    H, W, _ = Xgrid.shape
    if pos_cov_norm.shape[0] == 0:
        return np.zeros((H, W), dtype=np.float32)
    s = Xgrid.reshape(-1, 7)
    d2 = ((s[:, None, :] - pos_cov_norm[None, :, :]) ** 2).sum(axis=2)
    return (-np.sqrt(d2.min(axis=1))).reshape(H, W).astype(np.float32)


class Proxy:
    def __init__(self, ctx):
        self.ctx = ctx
        self.sm = np.zeros((ctx.H, ctx.W), dtype=np.float32)
        self._npos = -1

    def update(self, env):
        if len(env.found) != self._npos:
            self._npos = len(env.found)
            pn = self.ctx.norm_pos_cov(env.pos_cov)
            self.sm = proxy_score_map(self.ctx.Xgrid, pn)


def algo_greedy(env, rng, ctx):
    """Global greedy: at each step pick the unvisited cell with the highest proxy score (reused for H4 control)."""
    o = env.o
    pr = Proxy(ctx)
    while not env.done:
        pr.update(env)
        flat = np.argsort(pr.sm.ravel())[::-1]
        picked = None
        for idx in flat:
            i, j = idx // ctx.W, idx % ctx.W
            if (i, j) not in env.visited:
                picked = (i, j); break
        if picked is None:
            picked = (rng.randrange(ctx.H), rng.randrange(ctx.W))
        env.query(*picked)


def algo_hill_climb(env, rng, ctx):
    o = env.o
    H, W = ctx.H, ctx.W
    pr = Proxy(ctx)
    ci, cj = rng.randrange(H), rng.randrange(W)
    env.query(ci, cj)
    while not env.done:
        pr.update(env)
        sm = pr.sm
        best, best_s = None, sm[ci, cj]
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = ci + di, cj + dj
                if 0 <= ni < H and 0 <= nj < W and (ni, nj) not in env.visited:
                    if sm[ni, nj] > best_s:
                        best_s = sm[ni, nj]; best = (ni, nj)
        if best is None:
            flat = np.argsort(sm.ravel())[::-1]
            picked = None
            for idx in flat:
                i, j = idx // W, idx % W
                if (i, j) not in env.visited:
                    picked = (i, j); break
            if picked is None:
                picked = (rng.randrange(H), rng.randrange(W))
            ci, cj = picked
        else:
            ci, cj = best
        env.query(ci, cj)


def run_baseline(oracle, algo, seeds):
    ctx = EpisodeCtx(oracle)
    rows = []
    for seed in seeds:
        env = Env(oracle)
        rng = random.Random(seed)
        algo(env, rng, ctx)
        rows.append((seed, env.result()))
    return rows


# Main flow

def main():
    if LOG.exists():
        LOG.unlink()
    log("=== 4 harmful arms paired test ===")

    ablation = load_csv(ROOT / "results" / "ablation_cross" / "ablation_detail.csv")
    o3d5 = load_csv(ROOT / "results" / "ocean3d" / "regress_opt_F" / "framework_opt_o3d_O3D5_params7.csv")
    o3d6 = load_csv(ROOT / "results" / "ocean3d" / "regress_opt_F" / "framework_opt_o3d_O3D6_cluster.csv")
    o3d7 = load_csv(ROOT / "results" / "ocean3d" / "regress_opt_F" / "framework_opt_o3d_O3D7_adaptive.csv")

    out_rows = []

    def pick(sc, var):
        return [r for r in ablation if r["scenario"] == sc and r["variant"] == var]

    # H1: clouded_leopard full vs AUC_off (disable the entire density guidance chain)
    full_cl = pick("clouded_leopard", "full")
    off_cl = pick("clouded_leopard", "AUC_off")
    by_seed = {int(r["algo_seed"]): r for r in full_cl}
    a_pts = [by_seed[int(r["algo_seed"])]["points_found"] for r in off_cl]
    b_pts = [r["points_found"] for r in off_cl]
    a_q = [by_seed[int(r["algo_seed"])]["total_queries"] for r in off_cl]
    b_q = [r["total_queries"] for r in off_cl]
    # Harmful direction: disabling improves hits (AUC_off > full on points)
    out_rows.append(paired_stats("H1_density_guidance", "clouded_leopard", "points",
                                 a_pts, b_pts, "b_better",
                                 "harmful: disabling improves hits"))
    out_rows.append(paired_stats("H1_density_guidance", "clouded_leopard", "queries",
                                 a_q, b_q, "a_better",
                                 "queries: off saves more queries is the harmful direction"))

    # H2: chlorophyll full vs multicluster_off (queries drop sharply after disabling multicluster)
    full_ch = pick("chlorophyll", "full")
    off_ch = pick("chlorophyll", "multicluster_off")
    by_seed = {int(r["algo_seed"]): r for r in full_ch}
    a_pts = [by_seed[int(r["algo_seed"])]["points_found"] for r in off_ch]
    b_pts = [r["points_found"] for r in off_ch]
    a_q = [by_seed[int(r["algo_seed"])]["total_queries"] for r in off_ch]
    b_q = [r["total_queries"] for r in off_ch]
    out_rows.append(paired_stats("H2_multicluster", "chlorophyll", "points",
                                 a_pts, b_pts, "b_better",
                                 "harmful if off pts >= full (disabling loses no hits)"))
    out_rows.append(paired_stats("H2_multicluster", "chlorophyll", "queries",
                                 a_q, b_q, "a_better",
                                 "harmful: full is significantly more query-expensive"))

    # H3: 3-D hypoxic multicluster (O3D5 off vs O3D6/O3D7 on)
    for other, tag in [(o3d6, "O3D6"), (o3d7, "O3D7")]:
        by_seed = {int(r["seed"]): r for r in o3d5}
        a_pts = [by_seed[int(r["seed"])]["points_found"] for r in other]
        b_pts = [r["points_found"] for r in other]
        a_q = [by_seed[int(r["seed"])]["total_queries"] for r in other]
        b_q = [r["total_queries"] for r in other]
        out_rows.append(paired_stats(f"H3_multicluster_vs_{tag}", "o3d", "points",
                                     a_pts, b_pts, "a_better",
                                     "harmful if O3D5(off) pts >= cluster arm"))
        out_rows.append(paired_stats(f"H3_multicluster_vs_{tag}", "o3d", "queries",
                                     a_q, b_q, "b_better",
                                     "harmful if cluster arm is more query-expensive"))

    # H4: chlorophyll full vs greedy / hill_climb (rerun baselines)
    by_seed_ch = {int(r["algo_seed"]): r for r in pick("chlorophyll", "full")}
    oracle = SpeciesOracle(ROOT / "chlorophyll-search" / "data" / "oracle.npz")
    for algo, tag in [(algo_greedy, "greedy"), (algo_hill_climb, "hill_climb")]:
        rows = run_baseline(oracle, algo, SEEDS)
        bl = {s: r for s, r in rows}
        a_pts = [bl[s]["points_found"] for s in SEEDS]
        a_q = [bl[s]["total_queries"] for s in SEEDS]
        b_pts = [by_seed_ch[s]["points_found"] for s in SEEDS]
        b_q = [by_seed_ch[s]["total_queries"] for s in SEEDS]
        out_rows.append(paired_stats(f"H4_framework_vs_{tag}", "chlorophyll", "points",
                                     a_pts, b_pts, "a_better",
                                     "harmful if baseline pts >= framework"))
        out_rows.append(paired_stats(f"H4_framework_vs_{tag}", "chlorophyll", "queries",
                                     a_q, b_q, "b_better",
                                     "harmful: framework is significantly more query-expensive"))
        # Save baseline detail
        with open(HERE / f"baseline_chl_{tag}_detail.csv", "w", encoding="utf-8",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["algo", "seed", "points_found", "total_queries", "covered_cells"])
            for s in SEEDS:
                r = bl[s]
                w.writerow([tag, s, r["points_found"], r["total_queries"],
                            r["covered_cells"]])

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    log(f"[done] Output -> {OUT} ({len(out_rows)} rows)")
    log(f"[done] Baseline detail -> supplement_stats/baseline_chl_*.csv")


if __name__ == "__main__":
    main()
