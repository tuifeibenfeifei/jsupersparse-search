"""
_ablation_common.py
Common infrastructure shared by the Stage ablation scheduling scripts (Req. Sec. 15: unified setup for the scheduling and persistence layer).

Extracts the following shared logic to avoid ~150 duplicated lines in each of the 4 stage scripts:
    - YAML loading and single-config construction
    - calling DeepSeaExplorer.run_experiment for 20 seeds x 20 repeats
    - statistical aggregation (mean/std/success_rate)
    - CSV writing (unified columns, including the completion-based success rate)
    - comparison bar chart (with error bars, Req. Sec. 12)
    - convergence-curve plotting (Stage4-specific, Req. Sec. 10)
    - summary table printing

[!] This module contains no ablation-variable values; everything is driven by the YAML passed in by the caller (Req. Sec. 10).
"""

import sys
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Callable

import yaml
import matplotlib
matplotlib.use("Agg")   # enables plotting in headless environments
import matplotlib.pyplot as plt

# add the project root to the path so that `import src.*` works
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main_algorithm import DeepSeaExplorer
from src.cubiomes_wrapper import _DLL_PATH as CUBIOMES_DLL_PATH


def require_real_cubiomes() -> None:
    """Fail loudly instead of silently degrading to Mock mode.

    CubiomesWrapper falls back to pseudo-random feedback when cubiomes.dll is
    missing. That fallback is useful for smoke tests but produces meaningless
    numbers for a real experiment, so every experiment script calls this guard
    at start-up and a missing DLL becomes a hard error.
    """
    import os
    if not os.path.exists(CUBIOMES_DLL_PATH):
        raise SystemExit(
            "[FATAL] cubiomes.dll not found at {}.\n"
            "        Refusing to run: without the real generator the wrapper\n"
            "        silently falls back to Mock mode (pseudo-random feedback)\n"
            "        and every number produced would be meaningless.\n"
            "        Build Cubitect/cubiomes and place the DLL at src/cubiomes.dll."
            .format(CUBIOMES_DLL_PATH)
        )


# ==================== experiment constants (Req. Sec. 7 / Sec. 12) ====================
WORLD_SEEDS = list(range(1, 21))    # all experiments run on seeds 1~20
REPEATS = 20                        # repeat each seed 20 times and average
TOTAL_RUNS = len(WORLD_SEEDS) * REPEATS  # total runs per config (400)

RESULTS_DIR = PROJECT_ROOT / "results" / "ablation_logs"

# central config directory (used by scripts such as run_finder_showdown.py to locate YAML configs; Req. Sec. 10: config is the single source of values)
CONFIG_DIR = PROJECT_ROOT / "configs"

# unified CSV columns (success rate = mean deep-sea points / target deep-sea points, i.e. completion)
CSV_FIELDS = ["label", "mean_points", "std_points", "mean_queries",
              "std_queries", "success_rate", "target_reached_count", "n_runs"]


def load_yaml(path: Path) -> dict:
    """Safely load a YAML config (empty file returns an empty dict)."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_config(base_config: dict, overrides: dict, mode: str) -> dict:
    """
    Build a single config: the YAML is the single source of values (Req. Sec. 10); only inject the output mode and this config's
    sweep-variable overrides. No ablation values are hardcoded.
    """
    config = dict(base_config)
    config["output_mode"] = mode
    for key, value in overrides.items():
        config[key] = value
    return config


def run_single_config(config: dict, label: str, stage_title: str = "Stage") -> Dict[str, float]:
    """
    Run 20 world seeds x 20 repeats for a single config (Req. Sec. 12: separate environment/algorithm randomness).
    Returns: a stats dict containing mean/std/success_rate and the raw results (with label appended).
    """
    print(f"\n{'=' * 74}")
    print(f">>> {stage_title} ablation config: {label}")
    print(f"  world seeds {WORLD_SEEDS[0]}~{WORLD_SEEDS[-1]} x {REPEATS} repeats per seed "
          f"= {TOTAL_RUNS} runs")
    print(f"{'=' * 74}")

    explorer = DeepSeaExplorer(config)
    stats = explorer.run_experiment(
        world_seeds=WORLD_SEEDS,
        repeats=REPEATS,
        algo_seed_start=42,
    )
    stats["label"] = label
    return stats


def run_sweep(yaml_config: dict, sweep_field: str, mode: str,
              stage_title: str, label_prefix: str = "",
              target_key: Optional[str] = None) -> List[Dict[str, float]]:
    """
    Generic sweep: read the list field sweep_field from the YAML, build a single config for each value in the list and run it.
    The list field itself is removed from the shared config (Req. Sec. 9: the sweep file cannot be passed directly to Explorer).

    Args:
        yaml_config:  the loaded sweep YAML
        sweep_field:  the list key holding the values to sweep (e.g. "micro_strategy_list")
        mode:         simple / detailed
        stage_title:  stage name used in printing
        label_prefix: result label prefix (e.g. "finder=" / "threshold=")
        target_key:   config key into which the swept value is injected. By convention, when omitted, take sweep_field with the "_list"
                      suffix removed (micro_strategy_list -> micro_strategy). When the config key and the list key
                      differ in name (distance_threshold_list -> deep_sea_distance_threshold),
                      it must be passed explicitly.
    Returns:
        a list of one stats result per swept value
    """
    values = yaml_config.get(sweep_field, [])
    if not values:
        print(f"[skip] {sweep_field} is empty or missing")
        return []

    if target_key is None:
        target_key = sweep_field[:-5] if sweep_field.endswith("_list") else sweep_field

    shared = {k: v for k, v in yaml_config.items() if k != sweep_field}
    results = []
    for v in values:
        config = build_config(shared, {target_key: v}, mode)
        label = f"{label_prefix}{v}"
        results.append(run_single_config(config, label, stage_title))
    return results


def save_csv(results: List[Dict[str, float]], filename: str,
             out_dir: Optional[Path] = None) -> Optional[Path]:
    """Save stats results to CSV (Req. Sec. 14: store under the corresponding directory in results/)."""
    if not results:
        return None
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in CSV_FIELDS})
    print(f"[save] CSV written to: {filepath}")
    return filepath


def plot_comparison(results: List[Dict[str, float]], filename: str,
                    title: str, ylabel: str = "Average Deep Sea Points Found",
                    out_dir: Optional[Path] = None) -> Optional[Path]:
    """Generate a comparison bar chart with error bars (Req. Sec. 12: error bars show results are not accidental)."""
    if not results:
        return None
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename

    labels = [r["label"] for r in results]
    means = [r["mean_points"] for r in results]
    stds = [r["std_points"] for r in results]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 6))
    x_pos = list(range(len(labels)))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color="#2196F3", edgecolor="black")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.3,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"[save] chart written to: {filepath}")
    return filepath


def plot_convergence_curves(curves: Dict[str, List[Dict]], filename: str,
                            title: str = "Stage4: Convergence Curves",
                            out_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Plot a "queries - cumulative discovered points" convergence curve (Req. Sec. 10: Stage4 convergence-curve data).

    Args:
        curves: {config label: list of convergence_data}, each entry being
                {"query_count":int, "deep_sea_points":int, "stage":int}
    """
    if not curves:
        return None
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, data in curves.items():
        if not data:
            continue
        xs = [d["query_count"] for d in data]
        ys = [d["deep_sea_points"] for d in data]
        ax.plot(xs, ys, label=label, linewidth=1.5)
    ax.set_xlabel("Query Count")
    ax.set_ylabel("Cumulative Deep Sea Points")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"[save] convergence curve written to: {filepath}")
    return filepath


def save_convergence_csv(curves: Dict[str, List[Dict]], filename: str,
                         out_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Export the raw convergence-curve data to CSV (Req. Sec. 10: the standard "queries - cumulative discovered points" CSV).
    Long-table format: label, query_count, deep_sea_points, stage
    """
    if not curves:
        return None
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "query_count", "deep_sea_points", "stage"])
        for label, data in curves.items():
            for d in data:
                writer.writerow([label, d.get("query_count"),
                                 d.get("deep_sea_points"), d.get("stage")])
    print(f"[save] convergence CSV written to: {filepath}")
    return filepath


def print_summary(all_results: List[Dict[str, float]], stage_title: str,
                  next_hint: str = "") -> None:
    """Print the large ablation summary table (success rate = mean deep-sea points / target, completion-based)."""
    if not all_results:
        print(f"\n[fail] no valid config found")
        return
    print("\n" + "=" * 96)
    print(f"  {stage_title} ablation experiments all completed")
    print("=" * 96)
    print(f"  {'Config':<24} {'Mean points':>12} {'Std':>10} {'Mean queries':>10} "
          f"{'Success(completion)':>16} {'Times hit target':>14}")
    print("-" * 96)
    for r in all_results:
        target = r.get("target_deep_sea_points") or 40
        rate = r.get("success_rate")
        if rate is None:
            rate = (r["mean_points"] / target) if target else 0.0
        reached = r.get("target_reached_count", 0)
        n_runs = r.get("n_runs", 0)
        print(f"  {r['label']:<24} {r['mean_points']:>12.2f} {r['std_points']:>10.2f} "
              f"{r['mean_queries']:>10.1f} "
              f"{r['mean_points']:>7.2f}/{target} ({rate * 100:>5.1f}%) "
              f"{reached:>8}/{n_runs}")
    print("=" * 96)
    print("  success rate = mean deep-sea points / target deep-sea points (completion-based, for cross-config comparison)")
    if next_hint:
        print(f"\n  {next_hint}")


def make_arg_parser(description: str, part_choices: Optional[List[str]] = None):
    """Unified command-line arguments: --mode simple/detailed, --part run range."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--mode", choices=["simple", "detailed"], default="simple",
                        help="output mode: simple=summary (20-row averages), detailed=full (20 complete repeats + live query output)")
    if part_choices:
        parser.add_argument("--part", choices=part_choices, default="all",
                            help="run range")
    return parser


def print_stage_banner(stage_title: str, mode: str, ablation_vars: str):
    """Print the unified opening banner for all stage scripts."""
    print("=" * 74)
    print(f"  {stage_title}")
    print("=" * 74)
    print(f"  Scale       : world seeds {WORLD_SEEDS[0]}~{WORLD_SEEDS[-1]} "
          f"x {REPEATS} repeats per seed = {TOTAL_RUNS} runs per config")
    print(f"  Output mode : {'detailed (20 complete repeats + live query output)' if mode == 'detailed' else 'simple (20-row averages)'}")
    print(f"  Ablation vars: {ablation_vars}")
    print(f"  Output dir  : {RESULTS_DIR}")
    print("=" * 74)
