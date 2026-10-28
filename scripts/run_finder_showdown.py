"""
run_finder_showdown.py
Stage2 compact comparison: only ray_scan vs grid_scan (user-specified, to avoid the long runtime of all 5 strategies).

Purpose: decide which macro traversal strategy the final version should adopt.
Scale: 20 world seeds x 20 repeats = 400 runs per strategy (Req. Sec. 12).

WARNING: supports --finder argument to run a single strategy, so the two strategies can run in parallel to reduce wall-clock time:
       python run_finder_showdown.py --finder ray_scan
       python run_finder_showdown.py --finder grid_scan
   Without --finder, run both sequentially and output a summary comparison table.

All values come from configs/ablation/stage2_movement/finder_sweep.yaml (Req. Sec. 10:
config-driven, hardcoding ablation values is forbidden).
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _ablation_common as _C
from _ablation_common import (
    CONFIG_DIR, load_yaml, build_config, run_single_config,
    save_csv, plot_comparison, print_summary,
)

SHOWDOWN_YAML = "stage2_movement/finder_sweep.yaml"
CANDIDATE_FINDERS = ["ray_scan", "grid_scan"]


def run_one(finder: str, shared: dict, mode: str):
    """Run a single macro strategy, return stats dict"""
    config = build_config(shared, {"chunk_finder": finder}, mode)
    return run_single_config(config, f"finder={finder}", "Stage2-compact")


def main():
    _C.require_real_cubiomes()   # refuse to run on Mock (missing cubiomes.dll)
    parser = argparse.ArgumentParser(description="Stage2 compact comparison: ray_scan vs grid_scan")
    parser.add_argument("--mode", default="simple", choices=["simple", "detailed"],
                        help="Output mode: simple=20 average rows per config; detailed=full results + real-time queries")
    parser.add_argument("--finder", default=None, choices=CANDIDATE_FINDERS,
                        help="Run only the specified strategy (for parallel execution); default runs both sequentially")
    parser.add_argument("--no-plot", action="store_true", help="Output only CSV, do not generate PNG")
    args = parser.parse_args()

    yaml_cfg = load_yaml(CONFIG_DIR / SHOWDOWN_YAML)
    # The list key is scheduling metadata and cannot be passed directly to Explorer (Req. Sec. 9)
    shared = {k: v for k, v in yaml_cfg.items() if k != "chunk_finder_list"}

    finders = [args.finder] if args.finder else CANDIDATE_FINDERS

    if args.finder:
        # Single-strategy mode: for parallel invocation, each writes its own CSV to avoid race conditions
        stats = run_one(args.finder, shared, args.mode)
        save_csv([stats], f"stage2_{args.finder}_summary.csv")
        print_summary([stats], f"Stage2 single-strategy result: {args.finder}")
        return

    print("=" * 74)
    print("  Stage2 compact comparison: ray_scan vs grid_scan")
    print("=" * 74)
    print(f"  Output mode  : {'simple (20 average rows)' if args.mode == 'simple' else 'detailed (full results + real-time query output)'}")
    print(f"  Candidate strategies: {', '.join(finders)}")
    print(f"  Inherited baseline: micro={shared.get('micro_strategy')}, "
          f"threshold={shared.get('deep_sea_distance_threshold')}")
    print(f"  Temperature architecture: in_state={shared.get('temperature_in_state')}, "
          f"guidance={shared.get('temperature_guidance_enabled')}")
    print(f"  Query budget  : {shared.get('max_queries_per_episode')} / "
          f"target {shared.get('target_deep_sea_points')} points")

    results = [run_one(f, shared, args.mode) for f in finders]

    csv_path = save_csv(results, "stage2_finder_showdown_summary.csv")
    if not args.no_plot:
        plot_comparison(results, "stage2_finder_showdown_comparison.png",
                        "Stage2 Finder Showdown (ray_scan vs grid_scan)")
    print_summary(results, "Stage2 compact comparison (ray_scan vs grid_scan)")

    # Final version selection recommendation
    best = max(results, key=lambda s: s.get("mean_points", 0))
    print(f"\n  > Final version recommendation: {best['label']}  "
          f"(mean deep-sea points {best.get('mean_points', 0):.2f})")
    if csv_path:
        print(f"  > CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
