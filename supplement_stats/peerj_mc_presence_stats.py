# -*- coding: utf-8 -*-
"""Compute MC presence stats across 20 worlds."""
import sys
sys.path.insert(0, r"C:\Users\19264\Desktop\deep-sea-exploration\species-search")
from oracle import SpeciesOracle

R = r"C:\Users\19264\Desktop\deep-sea-exploration"
vals = []
for ws in range(1, 21):
    o = SpeciesOracle(R + rf"\minecraft-search\oracle_npz\mc_world_seed{ws}.npz")
    vals.append(int(o.presence.sum()))
mean = sum(vals) / len(vals)
import statistics
print("per-world presence:", vals)
print(f"mean={mean:.1f} sd={statistics.stdev(vals):.1f} min={min(vals)} max={max(vals)}")
