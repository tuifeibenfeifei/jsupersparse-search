"""
run_final_model.py
Final model run script (L5 final validation layer)

[Responsibility] Corresponds to Req. Sec. 14 run_final_model.py:
    Read configs/final_model.yaml (Stage1~4 optimal parameter combination) -> call
    main_algorithm.run_experiment() to run specified seeds x repeats -> collect results
    -> write to results/final_logs/ -> plot final analysis figures.

[Dual output versions] (same convention as each Stage script):
    simple   -> simple version: one row of averages per world seed (N rows, N=number of seeds)
    detailed -> detailed version: full results per run + real-time query output

[Command line]
    --mode    simple / detailed (default simple, for batch data generation)
    --seeds   seed range, e.g. 1-20 or 1,2,3 (default 1-20)
    --repeats number of repeats per seed (default 20)
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ablation_common as C

FINAL_MODEL_YAML = C.PROJECT_ROOT / "configs" / "final_model.yaml"
FINAL_RESULTS_DIR = C.PROJECT_ROOT / "results" / "final_logs"


def parse_seed_range(spec: str):
    """Parse seed range: '1-20' -> [1..20]; '1,3,5' -> [1,3,5]"""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s.strip()]


def main():
    C.require_real_cubiomes()   # refuse to run on Mock (missing cubiomes.dll)
    parser = argparse.ArgumentParser(description="Final model run script (L5 final validation layer)")
    parser.add_argument("--mode", choices=["simple", "detailed"], default="simple",
                        help="Output mode: simple=one average row per seed; detailed=full results per run + real-time queries")
    parser.add_argument("--seeds", default="1-20",
                        help="World seed range, e.g. 1-20 or 1,2,3 (default 1-20)")
    parser.add_argument("--repeats", type=int, default=20,
                        help="Number of repeats per world seed (default 20)")
    args = parser.parse_args()

    if not FINAL_MODEL_YAML.exists():
        print(f"[FAIL] Config file not found: {FINAL_MODEL_YAML}")
        print("Please first fill configs/final_model.yaml based on Stage1~4 ablation results")
        sys.exit(1)

    world_seeds = parse_seed_range(args.seeds)
    config = C.load_yaml(FINAL_MODEL_YAML)
    config["output_mode"] = args.mode

    C.print_stage_banner(
        "L5 final model validation", args.mode,
        f"Seeds {world_seeds[0]}~{world_seeds[-1]} x {args.repeats} repeats, "
        f"config source configs/final_model.yaml",
    )
    FINAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    from src.main_algorithm import DeepSeaExplorer

    explorer = DeepSeaExplorer(config)
    stats = explorer.run_experiment(
        world_seeds=world_seeds,
        repeats=args.repeats,
        algo_seed_start=42,
    )

    # Write final model stats CSV + convergence curve CSV/PNG (Req. Sec. 14: results/final_logs/)
    stats["label"] = "final_model"
    C.save_csv([stats], "final_model_summary.csv", out_dir=FINAL_RESULTS_DIR)

    raw = stats.get("raw_results", [])
    if raw:
        # Package convergence trajectory by config label (all runs combined into one average curve)
        curve_label = "final_model"
        curves = {curve_label: [
            {"query_count": d.get("query_count", 0),
             "deep_sea_points": d.get("deep_sea_points", 0),
             "stage": d.get("stage", 0)}
            for r in raw for d in r.get("convergence_data", [])
        ]}
        C.save_convergence_csv(curves, "final_model_convergence.csv",
                               out_dir=FINAL_RESULTS_DIR)
        C.plot_convergence_curves(curves, "final_model_convergence.png",
                                  title="Final Model: Convergence Curves "
                                        "(Query Count vs Deep Sea Points)",
                                  out_dir=FINAL_RESULTS_DIR)

    C.print_summary([stats], "L5 final model",
                    next_hint="Final run complete: see results/final_logs/")

    # Print final config key points (Req. Sec. 10: dynamic output header is already printed inside DeepSeaExplorer)
    print("\n  > Final config summary")
    print(f"    macro={config.get('chunk_finder')} | micro={config.get('micro_strategy')} "
          f"| threshold={config.get('deep_sea_distance_threshold')}")
    print(f"    temp={config.get('temperature_enabled')} | "
          f"guidance={config.get('temperature_guidance_enabled')} | "
          f"curiosity_weight={config.get('curiosity_weight')}")
    print(f"    budget={config.get('max_queries_per_episode')} queries / "
          f"target {config.get('target_deep_sea_points')} points")


if __name__ == "__main__":
    main()
