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
"""make_plots.py - rerun 3 seeds to collect plot data, produce 4 figures (English matplotlib).
Figures:
  1) True drift trajectory + discovered points + visited cells overlay (seed101 main, side-by-side with seed102)
  2) Cumulative discovered points vs query count (3 seeds)
  3) Bar chart of AUC-selected covariate weights
  4) Visitation heatmap (seed101)
"""
import sys
from pathlib import Path

ROOT = Path(REPO_ROOT)
sys.path.insert(0, str(ROOT / "species-search"))
sys.path.insert(0, str(ROOT / "maritime-search"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from species_agent import SpeciesSearcher            # noqa: E402
from build_maritime_oracle import MaritimeOracle     # noqa: E402
from run_maritime import CFG, SEEDS, NPZ             # noqa: E402

OUT = ROOT / "results" / "maritime_search"
oracle = MaritimeOracle(NPZ)

# ---- Rerun collection ----
visited, found, conv = {}, {}, {}
for seed in SEEDS:
    s = SpeciesSearcher(oracle, CFG)
    r = s.run_episode(seed)
    vg = np.zeros((oracle.nrows, oracle.ncols), dtype=np.uint8)
    for (i, j) in s.visited_cells:
        if oracle.in_bounds(i, j):
            vg[i, j] = 1
    visited[seed] = vg
    found[seed] = r["found_cells"]
    conv[seed] = r["convergence"]

cov_names = oracle.cov_names
traj = oracle.traj                       # (601,2) km [x,y]
P0 = oracle.P0_cell                      # (i,j)
cell = oracle.cell_km

# ================= Fig 1: Trajectory overlay (left seed101 / right seed102) =================
fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
for ax, seed in zip(axes, [101, 102]):
    # Visited cells (scatter, cell center in km)
    vi, vj = np.where(visited[seed] > 0)
    ax.scatter((vj + 0.5) * cell, (vi + 0.5) * cell, s=4, c="lightgray",
               alpha=0.5, label="Visited cells")
    # True drift trajectory
    ax.plot(traj[:, 0], traj[:, 1], "-", c="tab:blue", lw=1.8, label="True drift trajectory")
    ax.plot(traj[0, 0], traj[0, 1], "o", c="black", ms=8, label="Wreck point P0")
    ax.plot(traj[-1, 0], traj[-1, 1], "D", c="tab:blue", ms=7, label="Position after 300h")
    # Discovered points
    if found[seed]:
        fc = np.array(found[seed])
        ax.scatter((fc[:, 1] + 0.5) * cell, (fc[:, 0] + 0.5) * cell,
                   s=60, marker="*", c="red", edgecolors="k", zorder=5,
                   label=f"Discovered points n={len(fc)}")
    nf = len(found[seed])
    ax.set_title(f"seed {seed}: {nf} points found", fontsize=12)
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.set_xlim(0, 300); ax.set_ylim(0, 300)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
axes[0].legend(loc="upper left", fontsize=8)
fig.suptitle("Maritime search and rescue: true drift trajectory + UAV visited cells + discovered points (physics simulation)",
             fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "fig1_trajectory_overlay.png", dpi=130)
plt.close(fig)

# ================= Fig 2: Cumulative found vs query count =================
fig, ax = plt.subplots(figsize=(8, 5))
colors = {101: "tab:gray", 102: "tab:red", 103: "tab:green"}
for seed in SEEDS:
    q = [c[0] for c in conv[seed]]
    f = [c[1] for c in conv[seed]]
    ax.plot(q, f, "-o", ms=3, color=colors[seed], label=f"seed {seed}")
ax.axhline(15, ls="--", c="k", alpha=0.5, label="Target 15 points")
ax.set_xlabel("Query count (sorties, = time 0.5h/sortie)")
ax.set_ylabel("Cumulative discovered points")
ax.set_title("Cumulative discovered points vs query count (3 seeds)")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig2_cumulative_found.png", dpi=130)
plt.close(fig)

# ================= Fig 3: AUC-selected covariate weights =================
d = np.load(OUT / "_artifacts.npz", allow_pickle=True)
akeys = d["auc_keys"]; avals = d["auc_vals"]
fig, ax = plt.subplots(figsize=(9, 5))
labels = [cov_names[int(k)] for k in akeys]
order = np.argsort(avals)
bars = ax.barh([labels[i] for i in order], [avals[i] for i in order],
               color="teal")
for i, idx in enumerate(order):
    ax.text(avals[idx] + 0.005, i, f"{avals[idx]:.3f}", va="center", fontsize=9)
ax.set_xlabel("AUC sparse selection weight ((AUC-0.5)^2*4, mean across seeds)")
ax.set_title("AUC online-selected covariates (weights descending; only seed102 has enough positive samples to trigger)")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig3_auc_selected.png", dpi=130)
plt.close(fig)

# ================= Fig 4: Visitation heatmap (seed101) =================
fig, ax = plt.subplots(figsize=(7, 6))
hm = ax.imshow(visited[101], origin="lower", cmap="Blues",
               extent=[0, 300, 0, 300], alpha=0.6)
ax.plot(traj[:, 0], traj[:, 1], "-", c="tab:red", lw=1.8, label="True drift trajectory")
ax.plot(traj[0, 0], traj[0, 1], "o", c="black", ms=8, label="P0")
ax.set_title("seed101 visitation heatmap (blue=probed cells) - blind sweep grid and drift band did not intersect")
ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
ax.legend(loc="upper left", fontsize=9)
fig.colorbar(hm, ax=ax, label="Probed")
fig.tight_layout()
fig.savefig(OUT / "fig4_visited_heatmap.png", dpi=130)
plt.close(fig)

print("figures done ->", OUT)
print("seed points:", {s: len(found[s]) for s in SEEDS})
