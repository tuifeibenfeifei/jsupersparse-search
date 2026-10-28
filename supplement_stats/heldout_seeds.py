# -*- coding: utf-8 -*-
import os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve()
for _ in range(8):
    if (REPO_ROOT / 'configs').is_dir():
        break
    REPO_ROOT = REPO_ROOT.parent
"""held-out seed block validation (SEEDS=201-220, separate from tuning/evaluation block 101-120).

Scenario x variant: clouded_leopard / chlorophyll x {full, AUC_off, multicluster_off};
chlorophyll baselines rerun greedy/hill_climb (reusing harmful_arms Env/Proxy).
Outputs ablation_detail_heldout.csv / heldout_chl_baselines.csv / heldout_summary.csv.
Statistics consistent with manuscript TABLE XI: one-tailed Wilcoxon + Cliff's delta.
"""
import argparse
import csv
import math
import random
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
ABL.mkdir(parents=True, exist_ok=True)

HELDOUT_DETAIL = ABL / "ablation_detail_heldout.csv"
CHL_BASE_OUT = HERE / "heldout_chl_baselines.csv"
SUMMARY_OUT = HERE / "heldout_summary.csv"
LOG = HERE / "heldout_seeds.log"

SEEDS = list(range(201, 221))        # held-out block (20 seeds)
TUNED = list(range(101, 121))        # tuning/evaluation block (for control)

# Config identical to run_cross_ablation.py
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
VARIANTS = [
    ("full", {}),
    ("AUC_off", {"guidance_on": False, "sparse_guidance": False}),
    ("multicluster_off", {"multicluster": False}),
]
ORACLE_PATHS = {
    "clouded_leopard": ROOT / "species-search" / "data" / "clouded_leopard" / "oracle.npz",
    "chlorophyll": ROOT / "chlorophyll-search" / "data" / "oracle.npz",
}
CHL_ORACLE = ORACLE_PATHS["chlorophyll"]

TARGET_CHL = 20
MAX_Q_CHL = 4000
MIN_D = 1.0
STRUCT = list(range(7))


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def wilcox_onesided(a, b, direction):
    d = np.asarray(a, float) - np.asarray(b, float)
    if direction == "a_better":
        return stats.wilcoxon(d, zero_method="wilcox", alternative="greater").pvalue
    return stats.wilcoxon(d, zero_method="wilcox", alternative="less").pvalue


def cliff_delta(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("nan")
    sgn = sum(np.sum(b < x) - np.sum(b > x) for x in a)
    return sgn / (n * m)


def paired(a, b, direction):
    a = [float(x) for x in a]; b = [float(x) for x in b]
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    ma, mb = float(np.mean(a)), float(np.mean(b))
    p = wilcox_onesided(a, b, direction)
    return dict(n=n, mean_a=round(ma, 3), mean_b=round(mb, 3),
                wilcox_p=round(p, 4), cliff_delta=round(cliff_delta(a, b), 4))


# Chlorophyll baselines (reusing harmful_arms Env/Proxy/algo logic)
class Env:
    def __init__(self, oracle):
        self.o = oracle
        self.reset()

    def reset(self):
        self.o.reset_counter()
        self.found = []
        self.visited = set()
        self.pos_cov = []

    @property
    def q(self):
        return self.o.query_count

    @property
    def done(self):
        return self.q >= MAX_Q_CHL or len(self.found) >= TARGET_CHL

    def query(self, i, j):
        i = int(i); j = int(j)
        pres, cov = self.o.query_cell(i, j)
        if pres is None:
            return None
        self.visited.add((i, j))
        if pres:
            if all(math.hypot(i - fi, j - fj) >= MIN_D for fi, fj in self.found):
                self.found.append((i, j))
                self.pos_cov.append(cov.copy())
        return pres, cov

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


def load_csv(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    if LOG.exists():
        LOG.unlink()
    args = argparse.ArgumentParser()
    args.add_argument("--smoke", type=int, default=0, help="trial run N seeds per unit first")
    a = args.parse_args()
    smoke = a.smoke
    seeds = SEEDS[:smoke] if smoke else SEEDS

    log(f"=== held-out seed block validation (seeds {seeds[0]}-{seeds[-1]}, n={len(seeds)}) ===")

    # 1) Framework three configs (clouded_leopard / chlorophyll)
    all_rows = []
    t0 = time.time()
    for sc in ["clouded_leopard", "chlorophyll"]:
        oracle = SpeciesOracle(ORACLE_PATHS[sc])
        for variant, ov in VARIANTS:
            cfg = dict(BASE_CFG[sc], **ov)
            rows = []
            for ag in seeds:
                res = SpeciesSearcher(oracle, cfg).run_episode(ag)
                rows.append([sc, variant, 0, ag, res["points_found"],
                             res["total_queries"]])
            mp = sum(r[4] for r in rows) / len(rows)
            mq = sum(r[5] for r in rows) / len(rows)
            log(f"  {sc:16s} {variant:18s} pts={mp:.2f} q={mq:.0f} "
                f"({time.time()-t0:.1f}s elapsed)")
            all_rows.extend(rows)

    with open(HELDOUT_DETAIL, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "variant", "world_seed", "algo_seed",
                    "points_found", "total_queries"])
        w.writerows(all_rows)
    log(f"[1/3] Framework three configs -> {HELDOUT_DETAIL} ({(time.time()-t0)/60:.1f} min)")

    # 2) Chlorophyll greedy / hill_climb baselines (held-out seed block)
    oracle = SpeciesOracle(CHL_ORACLE)
    base_rows = []
    for algo, tag in [(algo_greedy, "greedy"), (algo_hill_climb, "hill_climb")]:
        for s, r in run_baseline(oracle, algo, seeds):
            base_rows.append([tag, s, r["points_found"], r["total_queries"],
                              r["covered_cells"]])
    with open(CHL_BASE_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algo", "seed", "points_found", "total_queries", "covered_cells"])
        w.writerows(base_rows)
    log(f"[2/3] Chlorophyll baselines -> {CHL_BASE_OUT}")

    # 3) Summary + paired test
    det = {sc: {} for sc in ["clouded_leopard", "chlorophyll"]}
    for r in all_rows:
        det[r[0]].setdefault(r[1], []).append(r)

    # Tuning block control (read original ablation_detail.csv)
    tuned = {sc: {v: [] for v, _ in VARIANTS} for sc in ["clouded_leopard", "chlorophyll"]}
    for r in load_csv(ABL / "ablation_detail.csv"):
        if r["scenario"] in tuned and r["variant"] in tuned[r["scenario"]]:
            tuned[r["scenario"]][r["variant"]].append(r)

    def blocks(sc, v, source):
        rows = source[sc][v]
        if rows and isinstance(rows[0], dict):
            return ([float(x["points_found"]) for x in rows],
                    [float(x["total_queries"]) for x in rows])
        return ([float(x[4]) for x in rows],
                [float(x[5]) for x in rows])

    summary = []
    log("\n--- Paired test (held-out block, n=%d) ---" % len(seeds))
    for sc in ["clouded_leopard", "chlorophyll"]:
        # (a) Harmful arm direction (same convention as harmful_arms_paired.py: a=full framework, b=control arm)
        full_p, full_q = blocks(sc, "full", det)
        auc_p, auc_q = blocks(sc, "AUC_off", det)
        mc_p, mc_q = blocks(sc, "multicluster_off", det)
        if sc == "clouded_leopard":
            s1 = paired(full_p, auc_p, "b_better")    # test full pts < AUC_off (off-hit higher = harmful)
            cmp_tag = "full_vs_AUC_off"
            summary.append(dict(block="heldout", scene=sc, comparison=cmp_tag,
                                metric="points", **s1))
        else:
            s1 = paired(full_q, mc_q, "a_better")     # test full queries > multicluster_off (off saves more = harmful)
            cmp_tag = "full_vs_multicluster_off"
            summary.append(dict(block="heldout", scene=sc, comparison=cmp_tag,
                                metric="queries", **s1))
        # Control: tuning block same convention
        t_full_p, t_full_q = blocks(sc, "full", tuned)
        t_auc_p, t_auc_q = blocks(sc, "AUC_off", tuned)
        t_mc_p, t_mc_q = blocks(sc, "multicluster_off", tuned)
        if sc == "clouded_leopard":
            ts = paired(t_full_p, t_auc_p, "b_better")
            tag = "tuned_full_vs_AUC_off"
            tmetric = "points"
        else:
            ts = paired(t_full_q, t_mc_q, "a_better")
            tag = "tuned_full_vs_multicluster_off"
            tmetric = "queries"
        summary.append(dict(block="tuned", scene=sc, comparison=tag,
                            metric=tmetric, **ts))
        log(f"  [{sc}] heldout {cmp_tag} p={s1['wilcox_p']} d={s1['cliff_delta']} "
            f"| tuned {ts['wilcox_p']} d={ts['cliff_delta']}")

    # (b) H4: chlorophyll framework vs greedy/hill (held-out block)
    fr_full = {int(r[3]): float(r[5]) for r in all_rows
               if r[0] == "chlorophyll" and r[1] == "full"}
    for tag in ("greedy", "hill_climb"):
        bl = {int(x[1]): float(x[3]) for x in base_rows if x[0] == tag}
        a_q = [bl[s] for s in seeds]
        b_q = [fr_full[s] for s in seeds]
        s = paired(a_q, b_q, "b_better")   # baseline queries fewer = framework harmful direction
        summary.append(dict(block="heldout", scene="chlorophyll",
                            comparison=f"full_vs_{tag}", metric="queries", **s))
        log(f"  [chlorophyll] full vs {tag} heldout p={s['wilcox_p']} d={s['cliff_delta']}")

    with open(SUMMARY_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    log(f"[3/3] Summary -> {SUMMARY_OUT}")
    log(f"=== Done ({(time.time()-t0)/60:.1f} min, seeds={seeds[0]}-{seeds[-1]}) ===")


if __name__ == "__main__":
    main()
