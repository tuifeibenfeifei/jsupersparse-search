# -*- coding: utf-8 -*-
"""Compute MC presence stats across 20 worlds."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve()
while not (REPO_ROOT / "configs").is_dir():
    if REPO_ROOT.parent == REPO_ROOT:
        raise RuntimeError("repository root (containing configs/) not found")
    REPO_ROOT = REPO_ROOT.parent
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT / "species-search"))
from oracle import SpeciesOracle

R = str(ROOT)
vals = []
for ws in range(1, 21):
    o = SpeciesOracle(R + rf"\minecraft-search\oracle_npz\mc_world_seed{ws}.npz")
    vals.append(int(o.presence.sum()))
mean = sum(vals) / len(vals)
import statistics
print("per-world presence:", vals)
print(f"mean={mean:.1f} sd={statistics.stdev(vals):.1f} min={min(vals)} max={max(vals)}")
