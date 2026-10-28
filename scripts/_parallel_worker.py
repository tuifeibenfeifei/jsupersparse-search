"""
_parallel_worker.py
Single-config execution subprocess for parallel ablation experiments (scheduled by run_stages_parallel.py; do not run standalone).

[Responsibility]
    Receive a job (config dict + label) → call DeepSeaExplorer.run_experiment()
    Run 20 seeds × 20 repetitions → write statistical results (raw_results excluded to control size) and Stage4
    convergence curve data to outdir/job_<idx>.pkl via pickle.

[Exit codes]
    0 = success (job_<idx>.pkl generated)
    1 = failure (error traceback written to job_<idx>.err; does not affect other jobs)
"""

import sys
import pickle
import argparse
import traceback
from pathlib import Path

# Project root + scripts directory added to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description="Parallel ablation single-config worker")
    parser.add_argument("--manifest", required=True, help="Job manifest (pickle) path")
    parser.add_argument("--job-idx", type=int, required=True, help="Job index")
    parser.add_argument("--outdir", required=True, help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest, "rb") as f:
        jobs = pickle.load(f)
    job = jobs[args.job_idx]
    label = job["label"]

    try:
        print(f"[worker {args.job_idx}] {label} starting", flush=True)

        import _ablation_common as C
        from src.main_algorithm import DeepSeaExplorer

        config = dict(job["config"])
        config["output_mode"] = "simple"   # Parallel runs uniformly use simple mode (one line of averages per seed)

        explorer = DeepSeaExplorer(config)
        stats = explorer.run_experiment(
            world_seeds=C.WORLD_SEEDS,
            repeats=C.REPEATS,
            algo_seed_start=42,
        )
        stats["label"] = label

        # Stage4 budget sweep requires convergence curves (same aggregation logic as original run_query_target_sweep script)
        curve = None
        if job.get("need_curve"):
            import scripts.run_ablation_stage4 as s4
            curve = s4._aggregate_convergence(stats.get("raw_results", []))

        # Remove raw_results (too large); aggregate statistics are sufficient to reconstruct all output files
        stats.pop("raw_results", None)

        with open(outdir / f"job_{args.job_idx}.pkl", "wb") as f:
            pickle.dump({"stats": stats, "curve": curve}, f)

        print(f"[worker {args.job_idx}] {label} done | "
              f"avg deep-sea points={stats.get('mean_points', 0):.2f} | "
              f"avg queries={stats.get('mean_queries', 0):.1f}", flush=True)

    except Exception:
        err_path = outdir / f"job_{args.job_idx}.err"
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print(f"[worker {args.job_idx}] {label} failed, see {err_path.name}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
