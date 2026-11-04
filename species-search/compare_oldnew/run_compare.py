# -*- coding: utf-8 -*-
"""run_compare.py - hard-version main run: two species x 20 seeds x (12 old/new algorithms + 2 framework configurations).
Protocol: target=80, min_dist=4.0, max_queries=60000, seeds=101..120.
Detail files land in results/species_search/compare_oldnew/compare_detail_{species}_hard.csv
Columns: algo/kind/seed/points_found/queries/covered_cells/covered_ratio/success/elapsed_s
Usage:
  python run_compare.py --smoke            # smoke test: 1 species, 1 seed, few algorithms + 1 framework, reduced budget
  python run_compare.py                    # full: two species x 20 seeds x all algorithms + framework
"""
from __future__ import annotations
import sys, time, argparse, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (make_oracle, run_one, ORACLE_PATHS, SEEDS,
                    TARGET, MAX_QUERIES, MIN_DIST)  # noqa: E402
from algos import AGENTS  # noqa: E402

SPECIES_DIR = HERE.parent
sys.path.insert(0, str(SPECIES_DIR))
from species_agent import SpeciesSearcher  # noqa: E402

OUT = SPECIES_DIR.parent / "results" / "species_search" / "compare_oldnew"
OUT.mkdir(parents=True, exist_ok=True)

# ---- framework configurations (reuse run_framework_sparse.BASE_CONFIG / run_adaptive_species.A11) ----
FW_BASE = dict(
    target_points=TARGET,
    max_queries=MAX_QUERIES,
    min_dist_cells=MIN_DIST,
    stage2_min=3,
    stage3_min=5,
    stage3_jump_interval=30,
    stage3_jump_min_ratio=1.25,
    stage3_jump_max_ratio=5.0,
    curiosity_weight=1.0,
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon_start=0.9,
    epsilon_end=0.1,
    epsilon_decay=0.999,
    stage1_grid_step=2,
    guidance_on=True,
    cov_in_state=True,
    recalc_every=20,
    stage2_topk=8,
    sparse_guidance=True,
    sparse_topk=6,
    sparse_min_pos=5,
)
# params2: covariate group [0,11] (bio1 + bio12), sparse_guidance=True
FW_PARAMS2 = dict(FW_BASE, cov_indices=[0, 11])
# A11 adaptive: adaptive=True + directed_jump=True, all 19 covariates
FW_A11 = dict(FW_BASE, adaptive=True, directed_jump=True, cov_indices=list(range(19)))

FW_CONFIGS = [
    ("framework_params2", "framework", FW_PARAMS2),
    ("framework_A11", "framework", FW_A11),
]


def run_framework_one(oracle, cfg, algo_name, kind, seed,
                      target=TARGET, max_queries=MAX_QUERIES):
    """Run one framework configuration and return a row with the same schema as the algorithm detail rows."""
    t0 = time.time()
    res = SpeciesSearcher(oracle, cfg).run_episode(seed)
    elapsed = time.time() - t0
    total = oracle.nrows * oracle.ncols
    return {
        "algo": algo_name,
        "kind": kind,
        "seed": seed,
        "points_found": int(res["points_found"]),
        "queries": int(res["total_queries"]),
        "covered_cells": int(res["covered_cells"]),
        "covered_ratio": float(res["covered_ratio"]),
        "success": bool(res["points_found"] >= target),
        "elapsed_s": round(elapsed, 2),
    }


def run_species(species: str, seeds, algos, fw_configs,
                target=TARGET, max_queries=MAX_QUERIES):
    oracle = make_oracle(species)
    rows = []
    for seed in seeds:
        for aname in algos:
            t0 = time.time()
            res = run_one(oracle, AGENTS[aname], seed,
                          target=target, max_queries=max_queries, min_dist=MIN_DIST)
            dt = time.time() - t0
            print(f"[{species}] seed={seed} algo={aname:22s} found={res['points_found']:2d} "
                  f"q={res['queries']:5d} cov={res['covered_cells']:5d} {dt:6.1f}s", flush=True)
            rows.append(res)
        for fw_name, fw_kind, fw_cfg in fw_configs:
            cfg = dict(fw_cfg, target_points=target, max_queries=max_queries,
                       min_dist_cells=MIN_DIST)
            res = run_framework_one(oracle, cfg, fw_name, fw_kind, seed,
                                    target=target, max_queries=max_queries)
            print(f"[{species}] seed={seed} algo={fw_name:22s} found={res['points_found']:2d} "
                  f"q={res['queries']:5d} cov={res['covered_cells']:5d} {res['elapsed_s']:6.1f}s",
                  flush=True)
            rows.append(res)
    df = pd.DataFrame(rows)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--species", default=None)
    args = ap.parse_args()

    if args.smoke:
        seeds = [101]
        species_list = [args.species or "black_muntjac"]
        algos = ["random", "grid", "go_explore", "gp_ucb", "rnd_novelty"]
        fw_configs = [FW_CONFIGS[0]]  # run params2 only, to validate the glue code
        target, max_queries = TARGET, 2000  # reduced budget for smoke testing, pipeline validation only
        tag = "smoke"
    else:
        seeds = SEEDS
        species_list = list(ORACLE_PATHS.keys())
        algos = list(AGENTS.keys())
        fw_configs = FW_CONFIGS
        target, max_queries = TARGET, MAX_QUERIES
        tag = "hard"

    for sp in species_list:
        print(f"===== {sp} (target={target}, min_dist={MIN_DIST}, budget={max_queries}) =====",
              flush=True)
        df = run_species(sp, seeds, algos, fw_configs,
                         target=target, max_queries=max_queries)
        outp = OUT / f"compare_detail_{sp}_{tag}.csv"
        df.to_csv(outp, index=False, encoding="utf-8-sig")
        print("saved", outp, flush=True)


if __name__ == "__main__":
    main()
