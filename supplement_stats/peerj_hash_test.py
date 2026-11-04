# -*- coding: utf-8 -*-
"""Hash-seed dependence test for CHL seed 103."""
import os
import subprocess
import sys
from pathlib import Path

PY = sys.executable

REPO_ROOT = Path(__file__).resolve()
while not (REPO_ROOT / "configs").is_dir():
    if REPO_ROOT.parent == REPO_ROOT:
        raise RuntimeError("repository root (containing configs/) not found")
    REPO_ROOT = REPO_ROOT.parent
ROOT = REPO_ROOT
REPO_S = repr(str(ROOT / "species-search"))
REPO_C = repr(str(ROOT / "chlorophyll-search"))
ORACLE = repr(str(ROOT / "chlorophyll-search" / "data" / "oracle.npz"))
OUT = str(Path(__file__).resolve().parent / "peerj_hash_test.txt")
CODE = (
    "import sys;"
    "sys.path.insert(0, REPO_S);"
    "sys.path.insert(0, REPO_C);"
    "from oracle import SpeciesOracle;"
    "from species_agent import SpeciesSearcher;"
    "o = SpeciesOracle(ORACLE);"
    "cfg = dict(target_points=20, max_queries=4000, min_dist_cells=1.0, stage2_min=3, stage3_min=5,"
    " stage3_jump_interval=30, stage3_jump_min_ratio=1.25, stage3_jump_max_ratio=5.0, curiosity_weight=1.0,"
    " learning_rate=0.1, discount_factor=0.95, epsilon_start=0.9, epsilon_end=0.1, epsilon_decay=0.999,"
    " stage1_grid_step=3, guidance_on=True, cov_in_state=True, recalc_every=20, stage2_topk=8,"
    " sparse_guidance=True, sparse_topk=6, sparse_min_pos=5, multicluster=True, cluster_ring_ratio=10.0,"
    " cluster_idle_limit=12, blank_jump=True, coverage_budget_ratio=0.12, blank_far_ratio=0.3,"
    " post_jump_grace=6, saturate_mask=False, cov_indices=list(range(19)), adaptive=True, directed_jump=True);"
    "r = SpeciesSearcher(o, cfg).run_episode(103);"
    "print(r['points_found'], r['total_queries'])"
)
with open(OUT, "w", encoding="utf-8") as f:
    for hs in ["0", "1", "2", "3", "4", "random", "random", "random"]:
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = hs
        code = (CODE.replace("REPO_S", REPO_S)
                .replace("REPO_C", REPO_C)
                .replace("ORACLE", ORACLE))
        r = subprocess.run([PY, "-c", code], capture_output=True, text=True, env=env, timeout=120)
        f.write(f"hashseed={hs}: {r.stdout.strip()} (rc={r.returncode})\n")
print("done ->", OUT)
