# -*- coding: utf-8 -*-
"""plot_mineral.py - four figures for the two-stage mineralized-outcrop experiment."""
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
OUT = ROOT / "results" / "mineral_search"
DATA = ROOT / "mineral-search" / "data"

vis = np.load(OUT / "_vis.npz", allow_pickle=True)
stagesA = np.load(DATA / "stageA_surface.npz")
presA = stagesA["presence"]
sidA = stagesA["true_struct_id"]
cov_names = list(vis["cov_names"])
SEEDS = [101, 102, 103]
TRUE_STRIKE = 75.0

# ---------- Figure 1: stage-A outcrop ground truth plus discoveries (seed 103 as the representative) ----------
fig, ax = plt.subplots(figsize=(7, 6.5))
# Background colour of the outcrop bands of the three structures
colors = {0: "#bdbdbd", 1: "#9ecae1", 2: "#fdae6b"}
for sid, c in colors.items():
    m = sidA == sid
    ax.scatter(np.where(m)[1], np.where(m)[0], s=3, c=c,
               label=f"true outcrop (structure {sid+1})", alpha=0.6)
# Discoveries (seed 103)
fA = vis["foundA_103"]
ax.scatter(fA[:, 1], fA[:, 0], s=40, facecolors="none", edgecolors="red",
           linewidths=1.5, label="stage-A discoveries (seed 103)", zorder=5)
# Decoded strike line
th = np.deg2rad(vis["strike_hat_103"])
cx, cy = 60, 60   # near the centre of block A
L = 55
dx, dy = L * np.sin(th), -L * np.cos(th)   # north is -i
ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], "r--", lw=1.5,
        label=f"decoded strike theta_hat={vis['strike_hat_103']:.1f} deg")
ax.plot([cx - L * np.sin(np.deg2rad(TRUE_STRIKE)), cx + L * np.sin(np.deg2rad(TRUE_STRIKE))],
        [cy + L * np.cos(np.deg2rad(TRUE_STRIKE)), cy - L * np.cos(np.deg2rad(TRUE_STRIKE))],
        "g-", lw=1.5, alpha=0.7, label=f"true strike {TRUE_STRIKE} deg")
ax.invert_yaxis()
ax.set_xlabel("j (east, km)"); ax.set_ylabel("i (north to south, km)")
ax.set_title("Fig. 1  Stage-A surface outcrop ground truth and discoveries (synthetic geological model)")
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
ax.set_aspect("equal")
fig.tight_layout(); fig.savefig(OUT / "fig1_stageA_outcrops.png", dpi=130); plt.close(fig)

# ---------- Figure 2: decoded strike vs true strike (angle comparison) ----------
fig, ax = plt.subplots(figsize=(7, 4.5), subplot_kw=dict(polar=True))
angles_true = np.deg2rad(TRUE_STRIKE)
ax.bar(angles_true, 1.0, width=0.25, color="green", alpha=0.5, label="true strike 75 deg")
for s, mk in zip(SEEDS, ["o", "s", "^"]):
    th = np.deg2rad(float(vis[f"strike_hat_{s}"]))
    err = float(vis[f"strike_err_{s}"])
    ax.plot([th], [1.0], marker=mk, markersize=12,
            label=f"seed {s} theta_hat={np.rad2deg(th):.1f} deg (error {err:.1f} deg)")
ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
ax.set_thetamin(0); ax.set_thetamax(180)
ax.set_yticks([])
ax.set_title("Fig. 2  Decoded main-structure strike theta_hat vs true strike (angle comparison)", va="bottom")
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig2_strike_decode.png", dpi=130); plt.close(fig)

# ---------- Figure 3: stage-B true ore body plus discoveries (3-D projected section) ----------
ore = vis["ore_mask"]   # (K,L,J)
# Take a vertical section along strike (j near 40) and project to 2-D: depth k vs lateral i
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
for ax, s in zip(axes, SEEDS):
    # Projection of the true ore body onto the (i, k) plane (any j)
    ore_proj = ore.any(axis=2)   # (K,L)
    ax.imshow(ore_proj.T, origin="lower", aspect="auto", cmap="Greys", alpha=0.4,
              extent=[0, ore.shape[1], 0, ore.shape[0] * 0.05],
              label="true ore sheet")
    fB = vis[f"foundB_{s}"]
    onst = vis[f"on_struct_{s}"]
    z = (fB[:, 0] + 0.5) * 0.05
    ax.scatter(fB[:, 1], z, c=["red" if o else "blue" for o in onst],
               s=35, edgecolors="k", linewidths=0.4, zorder=5)
    ax.set_title(f"seed {s} (theta_hat={float(vis[f'strike_hat_{s}']):.1f} deg)")
    ax.set_xlabel("lateral i (km)")
axes[0].set_ylabel("depth z (km)")
axes[0].scatter([], [], c="red", s=35, label="discoveries on the ore body")
axes[0].scatter([], [], c="blue", s=35, label="discoveries on weak anomalies / background")
axes[0].legend(loc="upper right", fontsize=8)
fig.suptitle("Fig. 3  Stage-B 3-D true ore sheet (grey) and projected discoveries (depth-lateral section)")
fig.tight_layout(); fig.savefig(OUT / "fig3_stageB_ore_projection.png", dpi=130); plt.close(fig)

# ---------- Figure 4: offline AUC of each stage-B covariate ----------
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(cov_names))
w = 0.26
for i, s in enumerate(SEEDS):
    aucs = vis[f"aucs_{s}"]
    ax.bar(x + (i - 1) * w, aucs, w, label=f"seed{s}")
ax.axhline(0.55, ls="--", c="gray", lw=1, label="AUC = 0.55 significance line")
ax.set_xticks(x); ax.set_xticklabels(cov_names, rotation=60, ha="right", fontsize=8)
ax.set_ylabel("Mann-Whitney AUC (true ore body vs background)")
ax.set_ylim(0.4, 1.02)
ax.set_title("Fig. 4  Stage-B covariate discrimination (offline full-grid AUC; cov1 = distance to the predicted structural plane)")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig4_b_auc_covariates.png", dpi=130); plt.close(fig)

print("figures saved:")
for p in ["fig1_stageA_outcrops.png", "fig2_strike_decode.png",
          "fig3_stageB_ore_projection.png", "fig4_b_auc_covariates.png"]:
    print(" ", OUT / p)
