# -*- coding: utf-8 -*-
"""run_timevary_v5.py — 五轮时变重测（物理仿真合成数据，tv_oracles）：
变体：
  moving_plume : framework_off / framework_decay300 / framework_fix_motion /
                 framework_fix_follow / framework_fix_follow_disperse
  abrupt_jump  : framework_decay300 / framework_fix_motion / framework_fix_disperse /
                 framework_fix_follow / framework_fix_follow_disperse
单进程串行（WORKERS=1）。2 场景 × 5 变体 × 10 seeds(101-110)。
输出到 v5/：per_seed_*.csv、summary.csv、progress.log；
abrupt_jump 附带 seed107/109 featurelog；moving_plume 附带 seed101/102 TraceSearcher 痕迹。
"""
import csv
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent            # results/timevary_search/v5
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT / "species-search"))
sys.path.insert(0, str(HERE.parent))

from species_agent import SpeciesSearcher          # noqa: E402
from tv_oracles import MovingPlumeOracle, AbruptJumpOracle  # noqa: E402

N = 60; MAX_Q = 1500; TARGET = 20; MIN_D = 2.0
SEEDS = list(range(101, 111))

FRAME = dict(target_points=TARGET, max_queries=MAX_Q, min_dist_cells=MIN_D,
             sparse_guidance=True, sparse_min_pos=5, sparse_topk=5, recalc_every=20)
DECAY = dict(time_decay=True, decay_mode="exp", decay_tau=300.0)
FIX_MOTION = dict(FRAME, **DECAY, mutation_detect=True, mutation_auto_motion=True)

SCENES = {
    "moving_plume": dict(
        cls=MovingPlumeOracle,
        variants={
            "framework_off":                dict(FRAME),
            "framework_decay300":           dict(FRAME, **DECAY),
            "framework_fix_motion":         dict(FIX_MOTION),
            "framework_fix_follow":         dict(FIX_MOTION, follow_mode=True),
            "framework_fix_follow_disperse": dict(FIX_MOTION, follow_mode=True,
                                                  mutation_disperse_jump=True),
        }),
    "abrupt_jump": dict(
        cls=AbruptJumpOracle,
        variants={
            "framework_decay300":           dict(FRAME, **DECAY),
            "framework_fix_motion":         dict(FIX_MOTION),
            "framework_fix_disperse":       dict(FIX_MOTION, mutation_disperse_jump=True),
            "framework_fix_follow":         dict(FIX_MOTION, follow_mode=True),
            "framework_fix_follow_disperse": dict(FIX_MOTION, follow_mode=True,
                                                  mutation_disperse_jump=True),
        }),
}

logf = open(HERE / "progress.log", "w", encoding="utf-8")


def log(msg):
    print(msg, flush=True)
    logf.write(msg + "\n")
    logf.flush()


def append_per_seed(scene, variant, row):
    p = HERE / ("per_seed_%s_%s.csv" % (scene, variant))
    write_header = not p.exists()
    with open(p, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "points_found", "total_queries", "reached"])
        if write_header:
            w.writeheader()
        w.writerow(row)


def dump_featurelog(scene, variant, seed, feat):
    p = HERE / ("featurelog_%s_%s_seed%d.csv" % (scene, variant, seed))
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_query", "n_pos", "kept_cov"])
        for (nq, npos, kept) in feat:
            w.writerow([nq, npos, ";".join(str(int(k)) for k, _ in kept)])


def main():
    t0 = time.time()
    for scene_name, scene_cfg in SCENES.items():
        for variant, cfg in scene_cfg["variants"].items():
            for seed in SEEDS:
                ts = time.time()
                try:
                    oracle = scene_cfg["cls"](nrows=N, ncols=N, sim_seed=1234)
                    res = SpeciesSearcher(oracle, cfg).run_episode(algo_seed=seed)
                except Exception:
                    log("[ERROR] %s/%s/seed%d" % (scene_name, variant, seed))
                    log(traceback.format_exc())
                    continue
                append_per_seed(scene_name, variant, dict(
                    seed=seed,
                    points_found=int(res["points_found"]),
                    total_queries=int(res["total_queries"]),
                    reached=int(res["points_found"] >= TARGET)))
                if scene_name == "abrupt_jump" and seed in (107, 109):
                    dump_featurelog(scene_name, variant, seed, res["feature_log"])
                log("[%s] %-28s seed=%d pts=%3d/%d q=%4d (%.1fs)"
                    % (scene_name, variant, seed, res["points_found"], TARGET,
                       res["total_queries"], time.time() - ts))
    # summary
    with open(HERE / "summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "variant", "n_seeds", "points_mean", "points_std",
                    "reach_rate", "queries_mean"])
        for scene_name, scene_cfg in SCENES.items():
            for variant in scene_cfg["variants"]:
                p = HERE / ("per_seed_%s_%s.csv" % (scene_name, variant))
                pts, qs, reach = [], [], []
                with open(p, encoding="utf-8") as rf:
                    for r in csv.DictReader(rf):
                        pts.append(int(r["points_found"]))
                        qs.append(int(r["total_queries"]))
                        reach.append(int(r["reached"]))
                pts = np.array(pts, float); qs = np.array(qs, float)
                w.writerow([scene_name, variant, len(pts),
                            round(pts.mean(), 2), round(pts.std(), 2),
                            round(float(np.mean(reach)), 3), round(qs.mean(), 1)])
    log("[done] 总耗时 %.1fs -> %s" % (time.time() - t0, HERE))


if __name__ == "__main__":
    main()
