"""
run_ablation_stage4.py
Stage4 hyperparameter fine-tuning layer ablation experiment scheduler script

[Responsibility] Corresponds to Req. Sec. 14 run_ablation_stage4.py:
    Read YAMLs under stage4_hyperparams/ (inheriting the first three stages' optimal) -> loop call
    main_algorithm runs 20 seeds x 20 times -> generate CSV and [convergence curves] -> save to results/ablation_logs/

[Ablation variables] Req. Sec. 8 Stage4 (hyperparameter fine-tuning layer):
    Lock the first three stages' optimal structure; only test [curiosity weight value] and [query count/target count].

[Req. Sec. 10 special] query_target_sweep must output a standard
    "query count - cumulative discovered points" CSV data when run, for plotting convergence curves.
    This script extracts each run's
    convergence trajectory from raw_results[].convergence_data returned by run_experiment, binned and averaged across 400 runs by query count to obtain a smooth convergence curve.
"""

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ablation_common as C

STAGE4_CONFIG_DIR = C.PROJECT_ROOT / "configs" / "ablation" / "stage4_hyperparams"


def run_curiosity_weight_sweep(mode: str) -> List[Dict[str, float]]:
    """Curiosity weight sweep [1,3,5], label like curiosity_weight=3"""
    path = STAGE4_CONFIG_DIR / "curiosity_weight_sweep.yaml"
    if not path.exists():
        print(f"[SKIP] config file not found: {path}")
        return []
    return C.run_sweep(
        C.load_yaml(path), sweep_field="curiosity_weight_list", mode=mode,
        stage_title="Stage4", label_prefix="curiosity_weight=",
        target_key="curiosity_weight",
    )


def _aggregate_convergence(raw_results: List[dict], n_bins: int = 100) -> List[Dict]:
    """
    Aggregate convergence trajectories from multiple runs into one smooth curve.

    Each run's convergence_data is [(query_count, deep_sea_points, stage), ...].
    Query step points differ across runs, so equal-width binning by query count (n_bins bins),
    average the (query_count, deep_sea_points) falling in each bin to obtain the cross-run average convergence trajectory.

    Returns: [{"query_count": bin center, "deep_sea_points": average cumulative points}, ...]
    """
    if not raw_results:
        return []

    # collect all (query_count, deep_sea_points) samples
    samples: List[tuple] = []
    max_q = 0
    for r in raw_results:
        for d in r.get("convergence_data", []):
            q = d.get("query_count", 0)
            p = d.get("deep_sea_points", 0)
            samples.append((q, p))
            if q > max_q:
                max_q = q
    if not samples or max_q <= 0:
        return []

    bin_width = max_q / n_bins
    sum_p = [0.0] * n_bins
    sum_q = [0.0] * n_bins
    count = [0] * n_bins
    for q, p in samples:
        idx = min(int(q / bin_width), n_bins - 1)
        sum_p[idx] += p
        sum_q[idx] += q
        count[idx] += 1

    curve = []
    for i in range(n_bins):
        if count[i] > 0:
            curve.append({
                "query_count": sum_q[i] / count[i],
                "deep_sea_points": sum_p[i] / count[i],
            })
    return curve


def run_query_target_sweep(mode: str):
    """
    Query budget/target paired sweep + convergence curve export (Req. Sec. 10 special).
    query_budget_list and target_list are paired by index, run pair by pair.
    Returns: (summary stats list, convergence curve dict {label: aggregated trajectory})
    """
    path = STAGE4_CONFIG_DIR / "query_target_sweep.yaml"
    if not path.exists():
        print(f"[SKIP] config file not found: {path}")
        return [], {}

    yaml_config = C.load_yaml(path)
    budgets = yaml_config.get("query_budget_list", [])
    targets = yaml_config.get("target_list", [])
    if not budgets:
        print("[SKIP] query_budget_list is empty")
        return [], {}
    # When target list is missing or length mismatches, uniformly use 40 as default target (Req. Sec. 13)
    if len(targets) != len(budgets):
        targets = [40] * len(budgets)

    shared = {k: v for k, v in yaml_config.items()
              if k not in ("query_budget_list", "target_list")}

    results: List[Dict[str, float]] = []
    curves: Dict[str, List[Dict]] = {}
    for budget, target in zip(budgets, targets):
        overrides = {"max_queries_per_episode": budget, "target_deep_sea_points": target}
        config = C.build_config(shared, overrides, mode)
        label = f"budget={budget}"
        stats = C.run_single_config(config, label=label, stage_title="Stage4")
        results.append(stats)
        # extract and aggregate convergence curves from raw results (Req. Sec. 10)
        curves[label] = _aggregate_convergence(stats.get("raw_results", []))

    return results, curves


def main():
    C.require_real_cubiomes()   # refuse to run on Mock (missing cubiomes.dll)
    args = C.make_arg_parser(
        "Stage4 hyperparameter fine-tuning layer ablation experiment scheduler script",
        part_choices=["all", "curiosity", "budget"],
    ).parse_args()
    mode = args.mode

    C.print_stage_banner(
        "Stage4 hyperparameter fine-tuning layer ablation experiment", mode,
        "Curiosity weight [1,3,5] + query budget sweep (convergence curves), inheriting first three stages' optimal",
    )

    curiosity_results: List[Dict[str, float]] = []
    budget_results: List[Dict[str, float]] = []
    curves: Dict[str, List[Dict]] = {}

    # 1. Curiosity weight sweep
    if args.part in ("all", "curiosity"):
        curiosity_results = run_curiosity_weight_sweep(mode)
        if curiosity_results:
            C.save_csv(curiosity_results, "stage4_curiosity_summary.csv")
            C.plot_comparison(curiosity_results, "stage4_curiosity_comparison.png",
                              "Stage4: Curiosity Weight Sweep [1,3,5]")

    # 2. Query budget sweep + convergence curves (Req. Sec. 10)
    if args.part in ("all", "budget"):
        budget_results, curves = run_query_target_sweep(mode)
        if budget_results:
            C.save_csv(budget_results, "stage4_budget_summary.csv")
            C.plot_comparison(budget_results, "stage4_budget_comparison.png",
                              "Stage4: Query Budget Sweep")
        if curves:
            # Req. Sec. 10: standard "query count - cumulative discovered points" CSV + convergence curve figure
            C.save_convergence_csv(curves, "stage4_convergence.csv")
            C.plot_convergence_curves(curves, "stage4_convergence_curves.png",
                                      "Stage4: Convergence Curves (Query Count vs Deep Sea Points)")

    # 3. Merge summary (Req. Sec. 14: stage4_summary.csv)
    all_results = curiosity_results + budget_results
    if all_results:
        C.save_csv(all_results, "stage4_summary.csv")
        C.print_summary(all_results, "Stage4",
                        next_hint="Next: determine final hyperparameters based on this table and convergence curves, "
                                  "fill all optimal Stage1~4 combinations into configs/final_model.yaml")
    else:
        print("\n[FAIL] No valid config found; please check configs/ablation/stage4_hyperparams/")
        sys.exit(1)


if __name__ == "__main__":
    main()
