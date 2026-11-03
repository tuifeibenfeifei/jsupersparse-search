# -*- coding: utf-8 -*-
"""make_pollution_plots.py - the four figures for the abrupt pollution tracing scene.
Reads results/pollution_search/_run_artifacts.npz and writes to results/pollution_search/.
matplotlib output in English.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve()
while not (REPO_ROOT / "configs").is_dir():
    if REPO_ROOT.parent == REPO_ROOT:
        raise RuntimeError("repository root (containing configs/) not found")
    REPO_ROOT = REPO_ROOT.parent
ROOT = REPO_ROOT
OUT = ROOT / "results" / "pollution_search"
ART = np.load(OUT / "_run_artifacts.npz", allow_pickle=True)

presence = ART["presence"].astype(bool)
sx, sy = int(ART["source_i"]), int(ART["source_j"])
cov_names = list(ART["cov_names"])
n_real = int(ART["n_real"][0])
seeds = list(ART["seeds"])
per = list(ART["per_seed"])

# ---------- Figure 1: true plume + detected points + visited cells (seed 101) ----------
rec0 = per[0]
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(presence, cmap="Greens", alpha=0.35, origin="upper",
          extent=[0, 120, 120, 0])
# visited cells
vis = rec0["visited"].astype(bool)
vy, vx = np.where(vis)
ax.scatter(vx + 0.5, vy + 0.5, s=4, c="steelblue", alpha=0.4, label="visited cells")
# detected points
fc = rec0["found_cells"]
ax.scatter(fc[:, 1] + 0.5, fc[:, 0] + 0.5, s=40, c="red", marker="^",
           edgecolors="darkred", label="detected above-threshold points", zorder=5)
# true source
ax.scatter([sy + 0.5], [sx + 0.5], s=260, c="gold", marker="*",
           edgecolors="black", linewidths=1.2, label="true pollution source", zorder=6)
ax.set_title(f"Fig. 1  True plume and search trajectory (seed={rec0['seed']})\n"
             f"green = above-threshold plume (7.0%)  blue dots = visited cells  red triangles = detected points  gold star = true source",
             fontsize=11)
ax.set_xlabel("eastward j (cells, 500 m each)")
ax.set_ylabel("northward i (cells)")
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim(0, 120); ax.set_ylim(120, 0)
fig.tight_layout()
fig.savefig(OUT / "fig1_plume_overlay_seed101.png", dpi=130)
plt.close(fig)

# ---------- Figure 2: cumulative detected points vs queries ----------
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#d62728", "#1f77b4", "#2ca02c"]
for rec, c in zip(per, colors):
    conv = rec["convergence"]
    ax.plot(conv[:, 0], conv[:, 1], marker=".", lw=1.8, color=c,
            label=f"seed={rec['seed']} ({int(rec['found_cells'].shape[0])} points at the end)")
ax.axhline(15, ls="--", c="gray", lw=1, label="target = 15 points")
ax.set_xlabel("queries (detection calls)")
ax.set_ylabel("cumulative detected above-threshold points")
ax.set_title("Fig. 2  Cumulative detected points vs queries (budget = 800)", fontsize=12)
ax.grid(alpha=0.3); ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig2_convergence.png", dpi=130)
plt.close(fig)

# ---------- Figure 3: AUC covariate-selection scores (mean over 3 seeds) ----------
auc_mat = np.zeros((len(seeds), 19))
for si, rec in enumerate(per):
    for k, v in rec["auc"].items():
        auc_mat[si, k] = v
auc_mean = auc_mat.mean(axis=0)
auc_std = auc_mat.std(axis=0)
order = np.argsort(auc_mean)[::-1]
fig, ax = plt.subplots(figsize=(10, 6))
bar_colors = []
for k in order:
    bar_colors.append("#d62728" if k < n_real else "#9aa0a6")
bars = ax.bar(range(19), auc_mean[order], yerr=auc_std[order],
              color=bar_colors, edgecolor="black", linewidth=0.5, capsize=3)
ax.axhline(0.55, ls="--", c="black", lw=1, label="AUC selection threshold = 0.55")
ax.set_xticks(range(19))
ax.set_xticklabels([cov_names[k] for k in order], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Mann-Whitney AUC (direction-free; 0.5 = no discrimination)")
ax.set_ylim(0.45, 1.05)
ax.set_title("Fig. 3  AUC separability of the 19 covariates (mean over 3 seeds)\nred = physically meaningful (8)  grey = pure noise (11)", fontsize=12)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig3_auc_features.png", dpi=130)
plt.close(fig)

# ---------- Figure 4: visit heat map (seed 101) ----------
fig, ax = plt.subplots(figsize=(8, 8))
heat = rec0["visited"].astype(float)
# overlay the plume outline
ax.imshow(presence, cmap="Greens", alpha=0.25, origin="upper",
          extent=[0, 120, 120, 0])
ax.imshow(heat, cmap="hot", alpha=0.7, origin="upper",
          extent=[0, 120, 120, 0], interpolation="nearest")
ax.scatter([sy + 0.5], [sx + 0.5], s=200, c="gold", marker="*",
           edgecolors="black", linewidths=1.0, zorder=6, label="true source")
ax.set_title(f"Fig. 4  Visit heat map (seed={rec0['seed']}, {int(heat.sum())} cells visited)\n"
             "hot = dense visiting  green = true plume", fontsize=11)
ax.set_xlabel("eastward j"); ax.set_ylabel("northward i")
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim(0, 120); ax.set_ylim(120, 0)
fig.tight_layout()
fig.savefig(OUT / "fig4_visited_heatmap_seed101.png", dpi=130)
plt.close(fig)

print("figures ->", OUT)
for f in ["fig1_plume_overlay_seed101.png", "fig2_convergence.png",
          "fig3_auc_features.png", "fig4_visited_heatmap_seed101.png"]:
    print("  ", f, (OUT / f).exists())
