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
"""make_drift_plots.py - maritime search and rescue reverse-drift 4 figures.
Read results/drift_search/_run_artifacts.npz, output to results/drift_search/. English matplotlib.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(REPO_ROOT)
OUT = ROOT / "results" / "drift_search"
ART = np.load(OUT / "_run_artifacts.npz", allow_pickle=True)

presenceA = ART["presenceA"].astype(bool)
presenceB = ART["presenceB"].astype(bool)
debris_cell = ART["debris_cell"]
P0_i, P0_j = int(ART["P0_i"]), int(ART["P0_j"])
N = int(ART["n"])
per = list(ART["per_seed"])
seeds = list(ART["seeds"])
cell = float(ART["cell_km"])

# Use the successful seed (103) for main figures
succ = [i for i, r in enumerate(per) if len(r["foundA"]) > 0]
main = succ[-1] if succ else 0
r = per[main]
main_seed = seeds[main]

# ---------- Fig 1: StageA debris true positions + discovered points overlay ----------
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(presenceA, cmap="Blues", alpha=0.4, origin="upper",
          extent=[0, N, N, 0])
# Debris ground truth
ax.scatter(debris_cell[:, 1] + 0.5, debris_cell[:, 0] + 0.5,
           s=60, c="navy", marker="X", label="Debris true positions (10 items)", zorder=5)
# A discovered points
fa = r["foundA"]
ax.scatter(fa[:, 1] + 0.5, fa[:, 0] + 0.5, s=70, c="red", marker="^",
           edgecolors="darkred", label=f"A discovered points ({len(fa)})", zorder=6)
# True P0
ax.scatter([P0_j + 0.5], [P0_i + 0.5], s=260, c="gold", marker="*",
           edgecolors="black", linewidths=1.2, label="True wreck point P0", zorder=7)
ax.set_title(f"Fig 1 StageA debris snapshot search (seed={main_seed})\n"
             f"Blue=frozen debris detection area ({int(presenceA.sum())} cells) Red triangle=A discovered points "
             f"Blue X=debris Gold star=P0", fontsize=11)
ax.set_xlabel("East j (cells, 2km/cell)"); ax.set_ylabel("North i (cells)")
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim(0, N); ax.set_ylim(N, 0)
fig.tight_layout()
fig.savefig(OUT / "fig1_stageA_debris_overlay.png", dpi=130)
plt.close(fig)

# ---------- Fig 2: Decoded candidate region (centroid +/- radius) vs true P0 ----------
centroid = r["centroid_km"]
err = r["err_km"]
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(presenceA, cmap="Blues", alpha=0.3, origin="upper", extent=[0, N, N, 0])
# Decoded centroid (cells)
cj = centroid[0] / cell - 0.5
ci = centroid[1] / cell - 0.5
ax.scatter([cj], [ci], s=120, c="green", marker="o", edgecolors="darkgreen",
           linewidths=1.5, label=f"Decoded centroid ({centroid[0]:.0f},{centroid[1]:.0f}km)", zorder=6)
# Uncertainty radius +/-10 cells = 20km
circ = plt.Circle((cj, ci), 10, color="green", fill=False, ls="--", lw=1.5,
                  label="Decoded uncertainty radius (+/-20km)")
ax.add_patch(circ)
ax.scatter([P0_j + 0.5], [P0_i + 0.5], s=260, c="gold", marker="*",
           edgecolors="black", linewidths=1.2, label=f"True P0 (error {err:.1f}km)", zorder=7)
ax.plot([cj, P0_j + 0.5], [ci, P0_i + 0.5], "g--", lw=1.2, zorder=5)
ax.set_title(f"Fig 2 StageA discovered points backward advection decode -> candidate wreck region vs true P0 (seed={main_seed})\n"
             f"Green circle=decoded centroid Green dashed circle=+/-20km uncertainty Gold star=true P0  centroid error={err:.1f}km", fontsize=11)
ax.set_xlabel("East j (cells)"); ax.set_ylabel("North i (cells)")
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim(0, N); ax.set_ylim(N, 0)
fig.tight_layout()
fig.savefig(OUT / "fig2_decode_centroid_vs_P0.png", dpi=130)
plt.close(fig)

# ---------- Fig 3: StageB discovered points vs true P0 ----------
fb = r["foundB"]
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(presenceB, cmap="Oranges", alpha=0.5, origin="upper", extent=[0, N, N, 0])
if len(fb):
    ax.scatter(fb[:, 1] + 0.5, fb[:, 0] + 0.5, s=80, c="red", marker="^",
               edgecolors="darkred", label=f"B discovered points ({len(fb)})", zorder=6)
ax.scatter([P0_j + 0.5], [P0_i + 0.5], s=260, c="gold", marker="*",
           edgecolors="black", linewidths=1.2, label=f"True P0 (B_to_P0={r['B_to_P0_km']:.1f}km)", zorder=7)
ax.set_title(f"Fig 3 StageB wreck point search (seed={main_seed})\n"
             f"Orange=P0 static evidence disk (R=3 cells) Red triangle=B discovered points Gold star=true P0", fontsize=11)
ax.set_xlabel("East j (cells)"); ax.set_ylabel("North i (cells)")
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim(0, N); ax.set_ylim(N, 0)
fig.tight_layout()
fig.savefig(OUT / "fig3_stageB_found_vs_P0.png", dpi=130)
plt.close(fig)

# ---------- Fig 4: StageB AUC covariate scores ----------
aucB = r["aucB"]
names = list(ART["cov_namesB"])
order = sorted(range(len(names)), key=lambda k: -aucB.get(k, 0.5))
vals = [aucB.get(k, 0.5) for k in order]
fig, ax = plt.subplots(figsize=(11, 6))
colors = ["#d62728" if k == 19 else "#9aa0a6" for k in order]
ax.bar(range(len(order)), vals, color=colors, edgecolor="black", linewidth=0.5)
ax.axhline(0.55, ls="--", c="black", lw=1, label="AUC selection threshold=0.55")
ax.set_xticks(range(len(order)))
ax.set_xticklabels([f"{k}:{names[k]}" for k in order], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Mann-Whitney AUC (direction-agnostic, 0.5=no discrimination)")
ax.set_ylim(0.45, 1.05)
ax.set_title(f"Fig 4 StageB 20-covariate AUC separability (seed={main_seed})\n"
             f"Red=20th covariate 'distance to decoded centroid' AUC={aucB.get(19,0.5):.3f} (highest)", fontsize=12)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig4_stageB_auc_features.png", dpi=130)
plt.close(fig)

print("figures ->", OUT)
for f in ["fig1_stageA_debris_overlay.png", "fig2_decode_centroid_vs_P0.png",
          "fig3_stageB_found_vs_P0.png", "fig4_stageB_auc_features.png"]:
    print("  ", f, (OUT / f).exists())
