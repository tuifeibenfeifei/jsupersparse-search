# -*- coding: utf-8 -*-
"""make_outputs.py - aggregate the detail CSVs (including the framework baselines) into a summary CSV and a comparison figure.
Framework baselines: per-seed CSVs from params2 + the A11 adaptive mean row (existing results, not re-run).
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RES = ROOT / "results" / "species_search"
OUT = RES / "compare_oldnew"

SPECIES_CN = {"black_muntjac": "Black muntjac", "clouded_leopard": "Clouded leopard"}

# Display order and display names of the algorithms
ORDER = ["random", "grid", "hill_climb", "simulated_annealing", "ga", "pso",
         "gp_ucb", "turbo", "go_explore", "map_elites", "rnd_novelty",
         "adaptive_sampling", "framework_params2", "framework_A11"]
CN = {
    "random": "Random sampling", "grid": "Grid sweep", "hill_climb": "Hill climbing",
    "simulated_annealing": "Simulated annealing", "ga": "Genetic algorithm", "pso": "Particle swarm",
    "gp_ucb": "GP-UCB", "turbo": "Turbo", "go_explore": "Go-Explore",
    "map_elites": "MAP-Elites", "rnd_novelty": "RND curiosity",
    "adaptive_sampling": "Adaptive sampling", "framework_params2": "Framework (params2)",
    "framework_A11": "Framework (A11 adaptive)",
}
KIND = {
    "random": "old", "grid": "old", "hill_climb": "old", "simulated_annealing": "old",
    "ga": "old", "pso": "old",
    "gp_ucb": "new", "turbo": "new", "go_explore": "new", "map_elites": "new",
    "rnd_novelty": "new", "adaptive_sampling": "new",
    "framework_params2": "framework", "framework_A11": "framework",
}


def load_framework(species):
    """Return [(algo, mean_points, mean_queries, mean_covered, success_rate)]."""
    rows = []
    # params2 per seed
    p = RES / f"framework_species_{species}_params2.csv"
    df = pd.read_csv(p)
    mp = df["points_found"].mean()
    mq = df["total_queries"].mean()
    mc = df["cov_area_cells"].mean()  # size of the framework covariate guidance area (not de-duplicated visits; used as a proxy)
    sr = (df["points_found"] >= 20).mean() * 100
    rows.append(("framework_params2", KIND["framework_params2"], round(mp, 2),
                 round(mq, 1), round(mc, 1), round(sr, 1)))
    # A11 adaptive mean row
    a = RES / "adaptive" / f"{species}_adaptive_summary.csv"
    ad = pd.read_csv(a)
    r = ad.iloc[0]
    rows.append(("framework_A11", KIND["framework_A11"],
                 round(float(r["mean_points"]), 2),
                 round(float(r["mean_queries"]), 1),
                 round(float(r["mean_covered"]), 1), np.nan))
    return rows


def build(species):
    detp = OUT / f"compare_detail_{species}_full.csv"
    if not detp.exists():
        raise FileNotFoundError(
            f"detail CSV not found: {detp}\n"
            "Run the comparison first, which needs the black-box oracle under species-search/data/:\n"
            f"  python species-search/compare_oldnew/run_compare.py --species {species}"
        )
    det = pd.read_csv(detp)
    # mean of the algorithms we ran
    g = det.groupby("algo").agg(
        mean_points=("points_found", "mean"),
        mean_queries=("queries", "mean"),
        mean_covered=("covered_cells", "mean"),
        n=("seed", "count"),
    ).reset_index()
    sr = det.groupby("algo")["points_found"].apply(lambda s: (s >= 20).mean() * 100).reset_index()
    sr.columns = ["algo", "success"]
    g = g.merge(sr, on="algo")

    fw = load_framework(species)
    fw_df = pd.DataFrame(fw, columns=["algo", "kind", "mean_points",
                                       "mean_queries", "mean_covered", "success"])
    g["kind"] = g["algo"].map(KIND)
    g = g[["algo", "kind", "mean_points", "mean_queries", "mean_covered", "success"]]
    all_df = pd.concat([g, fw_df], ignore_index=True)
    # sort
    all_df["ord"] = all_df["algo"].map({a: i for i, a in enumerate(ORDER)})
    all_df = all_df.sort_values("ord").drop(columns="ord").reset_index(drop=True)
    all_df["mean_points"] = all_df["mean_points"].round(2)
    all_df["mean_queries"] = all_df["mean_queries"].round(1)
    all_df["mean_covered"] = all_df["mean_covered"].round(1)
    all_df["success"] = all_df["success"].round(1)
    all_df.to_csv(OUT / f"compare_summary_{species}.csv", index=False, encoding="utf-8-sig")
    det.to_csv(OUT / f"compare_detail_{species}.csv", index=False, encoding="utf-8-sig")
    return all_df


def plot(species, df):
    labels = [CN.get(a, a) for a in df["algo"]]
    mp = df["mean_points"].values
    mq = df["mean_queries"].values
    colors = ["#999999" if k == "old" else "#4C72B0" if k == "new" else "#C44E52"
              for k in df["kind"]]
    fig, ax1 = plt.subplots(figsize=(13, 6))
    x = np.arange(len(labels))
    ax1.bar(x, mp, color=colors, alpha=0.85, label="Mean points found")
    ax1.set_ylabel("Mean points found (target=20)", color="#333333")
    ax1.set_ylim(0, 22)
    ax1.axhline(20, ls="--", color="green", lw=1, alpha=0.6)
    ax2 = ax1.twinx()
    ax2.plot(x, mq, "o-", color="#DD8452", label="Mean queries")
    ax2.set_ylabel("Mean queries (budget 20000)", color="#DD8452")
    ax2.set_ylim(0, 21000)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax1.set_title(f"Rare-species search algorithm comparison - {SPECIES_CN[species]}"
                  f" (20 seeds, target=20, budget 20000)")
    # legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend = [Patch(color="#999999", label="Old algorithms"),
              Patch(color="#4C72B0", label="New algorithms"),
              Patch(color="#C44E52", label="Framework baselines"),
              Line2D([0], [0], color="#DD8452", marker="o", label="Mean queries")]
    ax1.legend(handles=legend, loc="upper right", fontsize=8)
    fig.tight_layout()
    outp = OUT / f"compare_{species}.png"
    fig.savefig(outp, dpi=130)
    plt.close(fig)
    return outp


def main():
    summaries = {}
    for sp in ["black_muntjac", "clouded_leopard"]:
        df = build(sp)
        plot(sp, df)
        summaries[sp] = df
        print(df.to_string(index=False))
    return summaries


if __name__ == "__main__":
    main()
