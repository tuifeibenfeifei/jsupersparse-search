"""
run_ablation_stage2.py
Stage2 macro-movement layer ablation experiment scheduling script

[Responsibility] Corresponds to Req. Sec. 14 in requirements.txt run_ablation_stage2.py:
    Read YAML files under stage2_movement/ (inheriting Stage1 optimum) → loop calling main_algorithm
    Run 20 seeds × 20 repetitions → generate CSV and plots → store in results/ablation_logs/

[Ablation variables] Req. Sec. 8 Stage2 (macro-movement layer):
    Lock Stage1 optimal results; only test [macro-traversal strategy] and [deep-sea point distance threshold].

[Two experiment groups]
    1. finder_sweep   : comparison of 5 macro-traversal strategies (ray/spiral/adaptive/grid/anchor)
    2. dist_threshold : distance threshold [416,448,480,512] sweep (lock a single finder to avoid variable coupling)
"""

import sys
from pathlib import Path
from typing import Dict, List

# Reuse common scheduling module (YAML loading / run_experiment driver / CSV / plotting / summary)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ablation_common as C

STAGE2_CONFIG_DIR = C.PROJECT_ROOT / "configs" / "ablation" / "stage2_movement"


def run_finder_sweep(mode: str) -> List[Dict[str, float]]:
    """Macro-traversal strategy sweep (5 methods), label format finder=grid_scan"""
    path = STAGE2_CONFIG_DIR / "finder_sweep.yaml"
    if not path.exists():
        print(f"[Skip] Config file not found: {path}")
        return []
    return C.run_sweep(
        C.load_yaml(path), sweep_field="chunk_finder_list", mode=mode,
        stage_title="Stage2", label_prefix="finder=", target_key="chunk_finder",
    )


def run_distance_threshold_sweep(mode: str) -> List[Dict[str, float]]:
    """Distance threshold sweep [416,448,480,512], label format threshold=416"""
    path = STAGE2_CONFIG_DIR / "dist_threshold_sweep.yaml"
    if not path.exists():
        print(f"[Skip] Config file not found: {path}")
        return []
    return C.run_sweep(
        C.load_yaml(path), sweep_field="distance_threshold_list", mode=mode,
        stage_title="Stage2", label_prefix="threshold=",
        target_key="deep_sea_distance_threshold",
    )


def main():
    C.require_real_cubiomes()   # refuse to run on Mock (missing cubiomes.dll)
    args = C.make_arg_parser(
        "Stage2 macro-movement layer ablation experiment scheduling script",
        part_choices=["all", "finder", "threshold"],
    ).parse_args()
    mode = args.mode

    C.print_stage_banner(
        "Stage2 macro-movement layer ablation experiment", mode,
        "5 macro-traversal strategies + deep-sea point distance threshold [416,448,480,512] (inheriting Stage1 optimum)",
    )

    finder_results: List[Dict[str, float]] = []
    threshold_results: List[Dict[str, float]] = []

    # 1. Macro-traversal strategy sweep
    if args.part in ("all", "finder"):
        finder_results = run_finder_sweep(mode)
        if finder_results:
            C.save_csv(finder_results, "stage2_finder_summary.csv")
            C.plot_comparison(finder_results, "stage2_finder_comparison.png",
                              "Stage2: Macro-traversal Strategy Comparison (5 Methods)")

    # 2. Distance threshold sweep
    if args.part in ("all", "threshold"):
        threshold_results = run_distance_threshold_sweep(mode)
        if threshold_results:
            C.save_csv(threshold_results, "stage2_threshold_summary.csv")
            C.plot_comparison(threshold_results, "stage2_threshold_comparison.png",
                              "Stage2: Deep-sea Distance Threshold Sweep [416,448,480,512]")

    # 3. Merge summary (Req. Sec. 14: stage2_summary.csv)
    all_results = finder_results + threshold_results
    if all_results:
        C.save_csv(all_results, "stage2_summary.csv")
        C.print_summary(all_results, "Stage2",
                        next_hint="Next step: Determine the optimal Stage2 movement skeleton based on this table (traversal strategy + distance threshold), "
                                  "and fill it into the non-tested variables of the Stage3 config")
    else:
        print("\n[Failed] No valid configuration found. Please check configs/ablation/stage2_movement/")
        sys.exit(1)


if __name__ == "__main__":
    main()
