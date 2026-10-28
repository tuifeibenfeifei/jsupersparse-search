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
"""Plot budget-recall curve for MC scenario."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(REPO_ROOT)
csv_path = ROOT / "results" / "minecraft_search" / "budget_curve.csv"
out_png = ROOT / "figures" / "Figure_budget_recall.png"
out_png.parent.mkdir(parents=True, exist_ok=True)

budgets = [500, 1000, 2000, 5000, 10000]
data = {"framework": [], "random": [], "hill_climb": []}

with open(csv_path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for m in data:
    for b in budgets:
        vals = [float(r["points_found"]) for r in rows
                if r["method"] == m and int(r["budget"]) == b]
        data[m].append(np.mean(vals))

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
ax.plot(budgets, data["framework"], "o-", linewidth=2.5, markersize=8,
        color="#2196F3", label="Our framework (model-free)")
ax.plot(budgets, data["random"], "s--", linewidth=2, markersize=7,
        color="#FF9800", label="Random search")
ax.plot(budgets, data["hill_climb"], "^-.", linewidth=2, markersize=7,
        color="#9E9E9E", label="Hill climbing")

ax.set_xscale("log")
ax.set_xticks(budgets)
ax.set_xticklabels([str(b) for b in budgets])
ax.set_xlabel("Query budget (number of cells sampled)", fontsize=12)
ax.set_ylabel("Mean deep-ocean cells found (out of 40)", fontsize=12)
ax.set_title("Minecraft: Budget-Recall Curve", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 42)

# annotate low-budget gap
ax.annotate("+15% at B=500\n(sample efficiency)",
            xy=(500, 24.74), xytext=(700, 15),
            fontsize=9, color="#1565C0",
            arrowprops=dict(arrowstyle="->", color="#1565C0"))

plt.tight_layout()
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"saved -> {out_png}")
