"""
run_ablation_stage1.py
Stage1 perception-layer ablation experiment scheduling script

[Responsibilities] Strictly corresponds to the run_ablation_stage1.py entry in Req. Sec. 14 of the requirements:
    Data flow: read the YAML files under stage1_perception/ -> repeatedly call main_algorithm to run
              20 seeds x 20 repeats -> produce CSV and charts -> store under results/ablation_logs/
    Calls: calls main_algorithm.run_experiment() and matplotlib for plotting
    Outputs: stage1_summary.csv and stage1_comparison.png (determine the best perception method)

[Req. Sec. 10] Dual output versions:
    simple (output_mode=simple)   -> 20 rows of averages per config
    detailed (output_mode=detailed) -> 20 complete repeats + live query output

[Ablation variables] Req. Sec. 8 Stage1 (basic perception layer):
    Fix all other variables to defaults; only test [temperature on/off] and [9 micro-sampling methods].
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import matplotlib
matplotlib.use("Agg")   # enables plotting in headless environments
import matplotlib.pyplot as plt

# add the project root to the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.main_algorithm import DeepSeaExplorer

# ==================== experiment constants (Req. Sec. 7 / Sec. 12) ====================
WORLD_SEEDS = list(range(1, 21))    # all experiments run on seeds 1~20 (Req. Sec. 7)
REPEATS = 20                        # repeat each seed 20 times and average (Req. Sec. 12)
TOTAL_RUNS = len(WORLD_SEEDS) * REPEATS  # total runs per config (400)

STAGE1_CONFIG_DIR = PROJECT_ROOT / "configs" / "ablation" / "stage1_perception"
RESULTS_DIR = PROJECT_ROOT / "results" / "ablation_logs"

# simple/detailed switch, can be overridden by the --mode command-line argument
OUTPUT_MODE = "simple"


def load_yaml(path: Path) -> dict:
    """Safely load the YAML config."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_config(base_config: dict, overrides: dict, mode: str) -> dict:
    """
    Build a single config: the YAML content is the single source (Req. Sec. 10: config-file driven; no hardcoding),
    inject only the output mode. Do not override any ablation variable already declared in the YAML.
    """
    config = dict(base_config)
    config["output_mode"] = mode
    for key, value in overrides.items():
        config[key] = value
    return config


def run_single_config(config: dict, label: str) -> Dict[str, float]:
    """
    Run 20 world seeds x 20 repeats for a single config
    Returns: a stats dict {label, mean_points, std_points, mean_queries, std_queries, n_runs, ...}
    """
    print(f"\n{'=' * 74}")
    print(f">>> Stage1 ablation config: {label}")
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


def run_temperature_ablation(mode: str) -> List[Dict[str, float]]:
    """
    Run the temperature on/off control experiment
    Req. Sec. 8 Stage1: temperature perception ON as the black-box innovation baseline, OFF as the pure blind-sweep control
    """
    results = []
    for name in ["temp_on", "temp_off"]:
        config_path = STAGE1_CONFIG_DIR / f"{name}.yaml"
        if not config_path.exists():
            print(f"[skip] config file does not exist: {config_path}")
            continue
        yaml_config = load_yaml(config_path)
        # micro_strategy_list is a sweep-list field and is not passed to DeepSeaExplorer in single-config runs
        yaml_config.pop("micro_strategy_list", None)
        config = build_config(yaml_config, {}, mode)
        results.append(run_single_config(config, label=name))
    return results


def run_sampling_sweep(mode: str) -> List[Dict[str, float]]:
    """
    Run the sweep over 9 micro-sampling strategies
    Req. Sec. 9: sampling_sweep.yaml contains a list of 9 micro-strategies, iterated by this script
    """
    sweep_path = STAGE1_CONFIG_DIR / "sampling_sweep.yaml"
    if not sweep_path.exists():
        print(f"[skip] config file does not exist: {sweep_path}")
        return []

    sweep_config = load_yaml(sweep_path)
    strategy_list = sweep_config.get("micro_strategy_list", [])
    if not strategy_list:
        print("[skip] no micro_strategy_list list found in sampling_sweep.yaml")
        return []

    # exclude the list key; the rest are the locked baseline values shared by all strategies (Req. Sec. 8: non-tested variables are strictly locked)
    shared_config = {k: v for k, v in sweep_config.items() if k != "micro_strategy_list"}

    results = []
    for strategy in strategy_list:
        overrides = {"micro_strategy": strategy}
        config = build_config(shared_config, overrides, mode)
        results.append(run_single_config(config, label=strategy))
    return results


def save_csv(results: List[Dict[str, float]], filename: str) -> Optional[Path]:
    """Save stats results to CSV (Req. Sec. 14: store under results/ablation_logs/)."""
    if not results:
        return None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RESULTS_DIR / filename
    # success rate = mean deep-sea points / target deep-sea points (completion)
    # target_reached_count = the number of times the target was actually filled (auxiliary reference)
    fieldnames = ["label", "mean_points", "std_points", "mean_queries",
                  "std_queries", "success_rate", "target_reached_count", "n_runs"]

    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fieldnames})

    print(f"[save] CSV written to: {filepath}")
    return filepath


def plot_comparison(results: List[Dict[str, float]], filename: str,
                    title: str, ylabel: str) -> Optional[Path]:
    """Generate a comparison bar chart with error bars (Req. Sec. 12: error bars show results are not accidental)."""
    if not results:
        return None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RESULTS_DIR / filename

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


def parse_args():
    parser = argparse.ArgumentParser(description="Stage1 perception-layer ablation experiment scheduling script")
    parser.add_argument("--mode", choices=["simple", "detailed"], default=OUTPUT_MODE,
                        help="output mode: simple=summary (20-row averages), detailed=full (20 complete repeats + live query output)")
    parser.add_argument("--part", choices=["all", "temperature", "sampling"], default="all",
                        help="run range: all=everything, temperature=temperature on/off only, sampling=9 micro-strategies sweep only")
    return parser.parse_args()


def main():
    args = parse_args()
    mode = args.mode

    # refuse to run on Mock (missing cubiomes.dll) instead of silently fabricating data
    from src.cubiomes_wrapper import _DLL_PATH as _DLL
    if not os.path.exists(_DLL):
        raise SystemExit(
            "[FATAL] cubiomes.dll not found at {}.\n"
            "        Refusing to run: without the real generator the wrapper silently\n"
            "        falls back to Mock mode (pseudo-random feedback) and every number\n"
            "        produced would be meaningless.\n"
            "        Build Cubitect/cubiomes and place the DLL at src/cubiomes.dll."
            .format(_DLL))

    print("=" * 74)
    print("  Stage1 perception-layer ablation experiment")
    print("=" * 74)
    print(f"  Scale       : world seeds {WORLD_SEEDS[0]}~{WORLD_SEEDS[-1]} "
          f"x {REPEATS} repeats per seed = {TOTAL_RUNS} runs per config")
    print(f"  Output mode : {'detailed (20 complete repeats + live query output)' if mode == 'detailed' else 'simple (20-row averages)'}")
    print(f"  Ablation vars: temperature on/off + 9 micro-sampling methods (other variables locked to Stage1 baseline values)")
    print(f"  Output dir  : {RESULTS_DIR}")
    print("=" * 74)

    temp_results: List[Dict[str, float]] = []
    sampling_results: List[Dict[str, float]] = []

    # 1. temperature on/off control
    if args.part in ("all", "temperature"):
        temp_results = run_temperature_ablation(mode)
        if temp_results:
            save_csv(temp_results, "stage1_temperature_summary.csv")
            plot_comparison(
                temp_results,
                "stage1_temperature_comparison.png",
                "Stage1: Temperature ON vs OFF (Black-box Tactile Feedback)",
                "Average Deep Sea Points Found",
            )

    # 2. sweep over 9 micro-strategies
    if args.part in ("all", "sampling"):
        sampling_results = run_sampling_sweep(mode)
        if sampling_results:
            save_csv(sampling_results, "stage1_sampling_summary.csv")
            plot_comparison(
                sampling_results,
                "stage1_sampling_comparison.png",
                "Stage1: Micro-sampling Strategy Comparison (9 Methods)",
                "Average Deep Sea Points Found",
            )

    # 3. merge and summarize (Req. Sec. 14: stage1_summary.csv)
    all_results = temp_results + sampling_results
    if all_results:
        save_csv(all_results, "stage1_summary.csv")
        print("\n" + "=" * 96)
        print("  Stage1 ablation experiments all completed")
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
        print("\n  Next step: use this table to determine the best Stage1 perception method and fill it into Stage2's non-tested variables")
    else:
        print("\n[fail] no valid config found, please check configs/ablation/stage1_perception/")
        sys.exit(1)


if __name__ == "__main__":
    main()
