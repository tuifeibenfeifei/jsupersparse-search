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
"""finalize_baselines.py — Aggregate framework + 3 baselines, plot bar charts, write baseline_report.md.
Framework numbers are directly referenced from each regress_v2 CSV (no rerun)."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(REPO_ROOT / "results/baseline_compare")

# Framework numbers (source: each scenario regress_v2 summary CSV, seeds 101-110)
# (display name, target, framework mean_points, framework mean_queries, notes)
FRAMEWORK = {
    "pollution": ("Pollution source tracing", 15, 15.0, 185.4, "regress_v2 points/total_queries"),
    "maritime": ("Maritime search and rescue", 15, 1.0, 600.0, "regress_v2 points/total_queries"),
    "driftA": ("Reverse drift · Stage A", 6, 2.8, 397.5, "regress_v2 A_points/A_queries"),
    "mineralA": ("Mineralization outcrop · Stage A", 10, 10.0, 255.5, "regress_v2 A_points/A_queries"),
}
ORDER = ["pollution", "maritime", "driftA", "mineralA"]
METHODS = ["random", "grid", "hill_climb"]
M_LABEL = {"framework": "Proposed framework", "random": "Random baseline", "grid": "Grid baseline", "hill_climb": "Hill-climb baseline"}

# Read baseline summary
base = {}  # (skey, method) -> (mean_points, mean_queries, hit_rate)
with open(OUT / "baseline_summary.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        base[(r["scenario_key"], r["method"])] = (
            float(r["mean_points_found"]), float(r["mean_total_queries"]),
            float(r["hit_rate"]))

# ---- Merged comparison CSV ----
combined = OUT / "combined_compare.csv"
with open(combined, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["scenario_key", "scenario", "target", "method",
                "mean_points_found", "mean_total_queries", "hit_rate", "source"])
    for k in ORDER:
        sname, target, fp, fq, _ = FRAMEWORK[k]
        w.writerow([k, sname, target, "framework", fp, fq,
                    f"{fp/target:.3f}", "regress_v2 (already run, referenced)"])
        for m in METHODS:
            mp, mq, hr = base[(k, m)]
            w.writerow([k, sname, target, m, mp, mq, f"{hr:.3f}", "this baseline_compare"])

# ---- Per-scenario bar chart (4 methods points_found) ----
for k in ORDER:
    sname, target, fp, fq, _ = FRAMEWORK[k]
    names = ["framework"] + METHODS
    vals = [fp] + [base[(k, m)][0] for m in METHODS]
    qvals = [fq] + [base[(k, m)][1] for m in METHODS]
    labels = [M_LABEL[n] for n in names]
    colors = ["#2c7fb8", "#9ecae1", "#bdbdbd", "#fdae6b"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", lw=0.6)
    ax.axhline(target, ls="--", color="red", lw=1, label=f"target={target}")
    for b, v, q in zip(bars, vals, qvals):
        ax.text(b.get_x() + b.get_width()/2, v + target*0.02,
                f"{v:.1f}\n({q:.0f} queries)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean discovered points points_found")
    ax.set_title(f"{sname}: 4-method comparison (seeds 101-110 mean; bar labels show mean queries)")
    ax.set_ylim(0, target * 1.25)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"bar_{k}.png", dpi=130)
    plt.close(fig)

print("combined + bars done ->", combined)
