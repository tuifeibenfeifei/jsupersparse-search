"""
run_baselines.py
5-baseline-algorithm comparison experiment scheduling script (L4 baseline-comparison layer)

[Responsibilities] Corresponds to the run_baselines.py entry in Req. Sec. 14 of the requirements:
    read the 5 YAML files under configs/ablation/baselines/ -> repeatedly call each baseline algorithm's
    run_xxx(config, seed) -> collect results -> aggregate statistics -> write to results/baseline_logs/
    -> plot the comparison chart.

[Baseline algorithms] The 5 independent algorithms under src/baselines/ (depending only on L0):
    bfs / go_explore / icm / random_sampling / rnd
    Each algorithm's run function returns a result dict with the same structure as main_algorithm.run_single_episode().

[Randomness separation] Req. Sec. 12:
    The world seed controls the terrain (passed as the seed argument to run_xxx),
    and algorithm randomness is injected per run via an incrementing config["algo_seed"], ensuring that repeated runs under the same world seed
    produce different results (separating environment randomness from algorithm randomness).

[Command line] --algo select a subset / --seeds seed range / --repeats number of repeats
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ablation_common as C

BASELINE_CONFIG_DIR = C.PROJECT_ROOT / "configs" / "ablation" / "baselines"
BASELINE_RESULTS_DIR = C.PROJECT_ROOT / "results" / "baseline_logs"

# baseline algorithm registry: YAML filename (without extension) -> run function
BASELINES: Dict[str, Callable[[dict, int], dict]] = {
    "bfs": None,
    "go_explore": None,
    "icm": None,
    "random": None,
    "rnd": None,
}
_BASELINE_MODULES = {
    "bfs": ("src.baselines.bfs", "run_bfs"),
    "go_explore": ("src.baselines.go_explore", "run_go_explore"),
    "icm": ("src.baselines.icm", "run_icm"),
    "random": ("src.baselines.random_sampling", "run_random_sampling"),
    "rnd": ("src.baselines.rnd", "run_rnd"),
}


def _load_baseline_functions():
    """Lazily import the 5 baseline algorithms (avoid loading the numpy/network at script import time)."""
    for name, (module_name, func_name) in _BASELINE_MODULES.items():
        module = __import__(module_name, fromlist=[func_name])
        BASELINES[name] = getattr(module, func_name)


def run_single_baseline(algo_name: str, config: dict, world_seed: int,
                        algo_seed: int) -> dict:
    """
    Run one baseline episode.
    - config injects algo_seed (algorithm random seed); the world seed is passed as the seed argument to run_xxx.
    - returns a result dict (same structure as run_single_episode).
    """
    cfg = dict(config)
    cfg["algo_seed"] = algo_seed
    return BASELINES[algo_name](cfg, world_seed)


def run_baseline_config(algo_name: str, config: dict, world_seeds: List[int],
                        repeats: int, algo_seed_start: int = 42) -> Dict[str, float]:
    """
    Run world_seeds x repeats experiments for a single baseline algorithm and aggregate statistics.
    Returns: a stats dict with the same structure as run_experiment (with label).
    """
    all_points: List[int] = []
    all_queries: List[int] = []
    raw_results: List[dict] = []
    target = int(config.get("target_deep_sea_points", 40))
    algo_seed_counter = algo_seed_start

    print(f"\n{'=' * 74}")
    print(f">>> Baseline algorithm: {algo_name}")
    print(f"  world seeds {world_seeds[0]}~{world_seeds[-1]} x {repeats} repeats per seed "
          f"= {len(world_seeds) * repeats} runs")
    print(f"{'=' * 74}")

    for world_seed in world_seeds:
        seed_points: List[int] = []
        seed_queries: List[int] = []
        for _ in range(repeats):
            algo_seed = algo_seed_counter
            algo_seed_counter += 1
            try:
                result = run_single_baseline(algo_name, config, world_seed, algo_seed)
                points = result["deep_sea_points_found"]
                queries = result["total_queries"]
            except Exception as e:
                print(f"\n  [error] {algo_name} world_seed={world_seed} "
                      f"algo_seed={algo_seed} failed: {e}", file=sys.stderr)
                result = {"world_seed": world_seed, "algo_seed": algo_seed,
                          "deep_sea_points_found": 0, "total_queries": 0, "error": str(e)}
                points, queries = 0, 0

            seed_points.append(points)
            seed_queries.append(queries)
            all_points.append(points)
            all_queries.append(queries)
            result["world_seed"] = world_seed
            result["algo_seed"] = algo_seed
            raw_results.append(result)

        avg_p = sum(seed_points) / len(seed_points)
        avg_q = sum(seed_queries) / len(seed_queries)
        seed_rate = avg_p / target if target > 0 else 0.0
        print(f"  [simple] world seed {world_seed:>2} | mean deep-sea points={avg_p:>6.2f} | "
              f"mean queries={avg_q:>7.1f} | "
              f"success rate={avg_p:.2f}/{target} ({seed_rate * 100:>5.1f}%)")

    n = len(all_points)
    mean_points = sum(all_points) / n if n else 0.0
    mean_queries = sum(all_queries) / n if n else 0.0
    std_points = (sum((p - mean_points) ** 2 for p in all_points) / n) ** 0.5 if n else 0.0
    std_queries = (sum((q - mean_queries) ** 2 for q in all_queries) / n) ** 0.5 if n else 0.0
    success_rate = (mean_points / target) if target > 0 else 0.0
    target_reached_count = sum(1 for p in all_points if p >= target)

    stats = {
        "label": f"baseline={algo_name}",
        "mean_points": mean_points,
        "std_points": std_points,
        "mean_queries": mean_queries,
        "std_queries": std_queries,
        "n_runs": n,
        "success_rate": success_rate,
        "target_reached_count": target_reached_count,
        "target_deep_sea_points": target,
        "raw_results": raw_results,
    }

    print(f"  [overall] mean deep-sea points={mean_points:.2f}+/-{std_points:.2f} | "
          f"mean queries={mean_queries:.1f}+/-{std_queries:.1f} | "
          f"success rate={mean_points:.2f}/{target} ({success_rate * 100:.1f}%) | "
          f"times target filled={target_reached_count}/{n}")
    return stats


def main():
    C.require_real_cubiomes()   # refuse to run on Mock (missing cubiomes.dll)
    parser = argparse.ArgumentParser(description="5-baseline-algorithm comparison experiment scheduling script")
    parser.add_argument("--algo", default="all",
                        choices=["all"] + sorted(_BASELINE_MODULES.keys()),
                        help="run range: all = all 5 baselines, or a single named one")
    parser.add_argument("--seeds", default="1-20",
                        help="world seed range, e.g. 1-20 or 1,2,3 (default 1-20)")
    parser.add_argument("--repeats", type=int, default=20,
                        help="number of repeats per world seed (default 20)")
    args = parser.parse_args()

    world_seeds = parse_seed_range(args.seeds)
    C.print_stage_banner(
        "L4 baseline-algorithm comparison experiment", "simple",
        f"BFS / Go-Explore / ICM / Random Sampling / RND "
        f"(seeds {world_seeds[0]}~{world_seeds[-1]} x {args.repeats} repeats)",
    )
    BASELINE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _load_baseline_functions()

    algo_names = (sorted(_BASELINE_MODULES.keys())
                  if args.algo == "all" else [args.algo])

    all_stats: List[Dict[str, float]] = []
    for algo_name in algo_names:
        yaml_path = BASELINE_CONFIG_DIR / f"{algo_name}.yaml"
        if not yaml_path.exists():
            print(f"[skip] config file does not exist: {yaml_path}")
            continue
        config = C.load_yaml(yaml_path)
        stats = run_baseline_config(algo_name, config, world_seeds, args.repeats)
        all_stats.append(stats)

    if not all_stats:
        print("\n[fail] no valid baseline config found, please check configs/ablation/baselines/")
        sys.exit(1)

    C.save_csv(all_stats, "baselines_summary.csv", out_dir=BASELINE_RESULTS_DIR)
    C.plot_comparison(all_stats, "baselines_comparison.png",
                      "Baselines Comparison (BFS / Go-Explore / ICM / Random / RND)",
                      out_dir=BASELINE_RESULTS_DIR)
    C.print_summary(all_stats, "L4 baseline-algorithm comparison",
                    next_hint="Next: compare the final framework head-to-head against the above 5 baselines, "
                              "to demonstrate the sample-efficiency advantage of the black-box learning framework")


def parse_seed_range(spec: str) -> List[int]:
    """Parse a seed range: '1-20' -> [1..20]; '1,3,5' -> [1,3,5]."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s.strip()]


if __name__ == "__main__":
    main()
