# -*- coding: utf-8 -*-
"""Quick anchor check: reproduce MC world1 seeds 101-105 with current MC_CFG."""
import sys
sys.path.insert(0, r"C:\Users\19264\Desktop\deep-sea-exploration\species-search")
sys.path.insert(0, r"C:\Users\19264\Desktop\deep-sea-exploration\chlorophyll-search")
from oracle import SpeciesOracle
from species_agent import SpeciesSearcher

R = r"C:\Users\19264\Desktop\deep-sea-exploration"
o = SpeciesOracle(R + r"\minecraft-search\oracle_npz\mc_world_seed1.npz")
MC_CFG = dict(target_points=40, max_queries=10000, min_dist_cells=2.0, stage2_min=3, stage3_min=5,
              stage3_jump_interval=30, stage3_jump_min_ratio=1.25, stage3_jump_max_ratio=5.0,
              curiosity_weight=1.0, guidance_on=True, cov_in_state=True, cov_indices=[0, 1, 2, 3, 4],
              multicluster=True, cluster_ring_ratio=10.0, cluster_idle_limit=12, blank_jump=True,
              coverage_budget_ratio=0.12, time_decay=True, decay_mode="exp", decay_tau=300.0,
              mutation_detect=True, mutation_streak_thresh=40, follow_mode=True,
              mutation_disperse_jump=True, mutation_jump_interval=15)
out = R + r"\supplement_stats\peerj_mc_anchor_check.txt"
with open(out, "w", encoding="utf-8") as f:
    for s in [101, 102, 103, 104, 105]:
        r = SpeciesSearcher(o, MC_CFG).run_episode(s)
        f.write(f"{s} {r['points_found']} {r['total_queries']}\n")
print("done ->", out)
