# -*- coding: utf-8 -*-
# --- repository root (auto-located) ---
import os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve()
for _ in range(8):
    if (REPO_ROOT / 'configs').is_dir():
        break
    REPO_ROOT = REPO_ROOT.parent
# --------------------------------------
"""Task 2 plotting: Generate figures based on _artifacts.npz + baseline.csv produced by diagnose.py.
Figure a) Drift corridor + seed101 visited binary map + discovered points;
Figure b) Per-seed hit count vs query time curve.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = Path(REPO_ROOT / "results/maritime_search/bootstrap_analysis_v3")
art = np.load(OUT / "_artifacts.npz", allow_pickle=True)

traj = art["traj"]                 # (MAX_Q+1, 2) km [x, y]
cell_km = float(art["cell_km"])
R_km = float(art["R_km"])
nrows = int(art["nrows"]); ncols = int(art["ncols"])
P0_cell = tuple(int(v) for v in art["P0_cell"])
visited101 = art["visited101"]     # (nrows,ncols) 0/1
found101 = art["found101"]         # (K,2) row/col
seeds = [int(s) for s in art["seeds"]]

# ---- Figure a: drift corridor + visited101 + discovered points ----
fig, ax = plt.subplots(figsize=(9, 8))
# visited binary map (black = visited)
ax.imshow(visited101, cmap="Greys", origin="lower",
          extent=[0, ncols, 0, nrows], alpha=0.55, aspect="equal")
# drift trajectory km -> grid coordinates
tj = traj[:, 0] / cell_km - 0.5
ti = traj[:, 1] / cell_km - 0.5
ax.plot(tj, ti, color="tab:red", lw=1.8, label="target drift trajectory (corridor)")
# Start/end point
ax.scatter([tj[0]], [ti[0]], color="tab:green", s=80, zorder=5, marker="o", label="P0 water entry point")
ax.scatter([tj[-1]], [ti[-1]], color="tab:orange", s=80, zorder=5, marker="*", label="trajectory endpoint")
# Detection radius band (along trajectory ±R_cell)
R_cell = R_km / cell_km
ax.fill_between(tj, ti - R_cell, ti + R_cell, color="tab:red", alpha=0.12,
                label=f"detection radius band ±{R_cell:.0f} cells (={R_km:.0f}km)")
# Discovered points
if len(found101) > 0:
    ax.scatter(found101[:, 1], found101[:, 0], color="cyan", edgecolor="k",
               s=70, zorder=6, marker="X", label=f"seed101 discovered points (n={len(found101)})")
ax.set_xlim(0, ncols); ax.set_ylim(0, nrows)
ax.set_xlabel("column j (cells)"); ax.set_ylabel("row i (cells)")
ax.set_title("Maritime search seed101: Stage1 blind sweep column bands vs drift corridor geometry")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "fig_a_trajectory_overlay.png", dpi=130)
plt.close()
print("fig_a -> fig_a_trajectory_overlay.png", flush=True)

# ---- Figure b: per-seed hit count over time ----
fig, ax = plt.subplots(figsize=(10, 6))
for seed in seeds:
    q = art[f"q_{seed}"]; f = art[f"f_{seed}"]
    ax.plot(q, f, lw=1.4, marker=".", ms=4, label=f"seed{seed}")
ax.axhline(15, color="k", ls="--", lw=1, alpha=0.5, label="target=15")
ax.set_xlabel("query time (=time step, 0.5h per step)")
ax.set_ylabel("cumulative discovered points")
ax.set_title("Per-seed hit count over time (600 budget saturated vs drifting target)")
ax.legend(ncol=5, fontsize=8, loc="upper center")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "fig_b_convergence_curves.png", dpi=130)
plt.close()
print("fig_b -> fig_b_convergence_curves.png", flush=True)
print("PLOT DONE", flush=True)
