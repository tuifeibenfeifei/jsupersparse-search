"""
run_stages_parallel.py
Run all Stage1~4 ablation experiments in parallel to completion (equivalent to running run_ablation_stage1~4.py in sequence,
but scheduled in parallel as "one config = one subprocess", greatly reducing total wall time).

[Design]
    - 31 config jobs (Stage1: 2 temperatures + 9 strategies / Stage2: 5 sweeps + 4 thresholds /
      Stage3: 4 rewards / Stage4: 3 weights + 4 budgets); each job = 20 seeds x 20 repeats = 400 runs
    - job construction logic is identical to each stage script (same YAML, same label, same injection key)
    - run with N concurrent subprocesses (--workers, default 12); a single job crashing does not affect others
    - after all complete, rebuild the output files according to each stage script's naming and plotting logic
      (stageX_*.csv / stageX_*.png / stageX_summary.csv), fully equivalent to running serially

[Command line]
    --workers N    number of concurrent subprocesses (default 12; the machine has 16 threads, recommended 8~14)
    --python CMD   Python interpreter command (default "py -3.9")
    --outdir DIR   directory for intermediate artifacts and logs (default results/parallel_jobs/)
    --stage all|1|2|3|4   run only the given stage (default all)

[Viewing progress]
    outdir/progress.txt -- live progress (done/total, running jobs, failed jobs)
    outdir/job_<i>.log   -- run log for each job
"""

import sys
import time
import pickle
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import _ablation_common as C

WORKER = Path(__file__).resolve().parent / "_parallel_worker.py"
ABLATION_DIR = PROJECT_ROOT / "configs" / "ablation"

# the 4 Stage3 reward configs (identical to REWARD_CONFIGS in run_ablation_stage3.py)
STAGE3_REWARD_CONFIGS = [
    ("reward_all_on", "all_on (all-on control)"),
    ("reward_no_deep", "no_deep (deep-sea off)"),
    ("reward_no_curiosity", "no_curiosity (curiosity off)"),
    ("reward_no_penalty", "no_penalty (penalty off)"),
]

# output-file rebuild info per stage: group -> (output filename list, plot info)
# plot info: (png filename, title)


def build_jobs(only_stage: Optional[str] = None) -> List[dict]:
    """Build all jobs (config-construction logic identical to each stage script)."""
    jobs: List[dict] = []

    def add(group, yaml_rel, overrides, label, need_curve=False, pop_fields=()):
        cfg = dict(C.load_yaml(ABLATION_DIR / yaml_rel))
        for f in pop_fields:
            cfg.pop(f, None)
        for k, v in overrides.items():
            cfg[k] = v
        jobs.append({
            "group": group,
            "config": cfg,
            "label": label,
            "need_curve": need_curve,
        })

    # ---------------- Stage1 perception layer ----------------
    if only_stage in (None, "1"):
        # temperature on/off (pop the sweep-list key, same as the original script)
        for name in ["temp_on", "temp_off"]:
            add("stage1_temp", f"stage1_perception/{name}.yaml",
                {}, name, pop_fields=("micro_strategy_list",))
        # sweep over 9 micro-strategies
        sweep = C.load_yaml(ABLATION_DIR / "stage1_perception/sampling_sweep.yaml")
        shared = {k: v for k, v in sweep.items() if k != "micro_strategy_list"}
        for s in sweep.get("micro_strategy_list", []):
            jobs.append({"group": "stage1_sampling",
                         "config": {**shared, "micro_strategy": s},
                         "label": s, "need_curve": False})

    # ---------------- Stage2 movement layer ----------------
    if only_stage in (None, "2"):
        sweep = C.load_yaml(ABLATION_DIR / "stage2_movement/finder_sweep.yaml")
        shared = {k: v for k, v in sweep.items() if k != "chunk_finder_list"}
        for f in sweep.get("chunk_finder_list", []):
            jobs.append({"group": "stage2_finder",
                         "config": {**shared, "chunk_finder": f},
                         "label": f"finder={f}", "need_curve": False})

        sweep = C.load_yaml(ABLATION_DIR / "stage2_movement/dist_threshold_sweep.yaml")
        shared = {k: v for k, v in sweep.items() if k != "distance_threshold_list"}
        for t in sweep.get("distance_threshold_list", []):
            jobs.append({"group": "stage2_threshold",
                         "config": {**shared, "deep_sea_distance_threshold": t},
                         "label": f"threshold={t}", "need_curve": False})

    # ---------------- Stage3 reward layer ----------------
    if only_stage in (None, "3"):
        for cfg_name, label in STAGE3_REWARD_CONFIGS:
            add("stage3_reward", f"stage3_reward/{cfg_name}.yaml", {}, label)

    # ---------------- Stage4 hyperparameter layer ----------------
    if only_stage in (None, "4"):
        sweep = C.load_yaml(ABLATION_DIR / "stage4_hyperparams/curiosity_weight_sweep.yaml")
        shared = {k: v for k, v in sweep.items() if k != "curiosity_weight_list"}
        for w in sweep.get("curiosity_weight_list", []):
            jobs.append({"group": "stage4_curiosity",
                         "config": {**shared, "curiosity_weight": w},
                         "label": f"curiosity_weight={w}", "need_curve": False})

        sweep = C.load_yaml(ABLATION_DIR / "stage4_hyperparams/query_target_sweep.yaml")
        shared = {k: v for k, v in sweep.items()
                  if k not in ("query_budget_list", "target_list")}
        budgets = sweep.get("query_budget_list", [])
        targets = sweep.get("target_list", [])
        if len(targets) != len(budgets):
            targets = [40] * len(budgets)
        for b, t in zip(budgets, targets):
            jobs.append({"group": "stage4_budget",
                         "config": {**shared, "max_queries_per_episode": b,
                                    "target_deep_sea_points": t},
                         "label": f"budget={b}", "need_curve": True})

    return jobs


def run_jobs(jobs: List[dict], outdir: Path, workers: int, py_cmd: str) -> tuple:
    """Run all jobs with workers concurrent subprocesses; return (set of done indices, list of failed indices)."""
    pending = list(range(len(jobs)))
    running: Dict[int, subprocess.Popen] = {}
    done: set = set()
    failed: List[int] = []
    progress = outdir / "progress.txt"
    last_progress_write = 0.0

    with open(progress, "w", encoding="utf-8") as pf:
        pf.write(f"{len(jobs)} config jobs total, {workers} concurrent\n")

    while pending or running:
        # launch new jobs (water-level control; jobs with existing results are skipped = resume / checkpoint recovery)
        while len(running) < workers and pending:
            idx = pending.pop(0)
            if (outdir / f"job_{idx}.pkl").exists():
                done.add(idx)
                print(f"[schedule] job {idx} ({jobs[idx]['label']}) already has results, skipping", flush=True)
                continue
            log_path = outdir / f"job_{idx}.log"
            log_file = open(log_path, "a", encoding="utf-8")
            print(f"[schedule] launching job {idx} ({jobs[idx]['label']})", flush=True)
            running[idx] = subprocess.Popen(
                [py_cmd, str(WORKER), "--manifest", str(outdir / "manifest.pkl"),
                 "--job-idx", str(idx), "--outdir", str(outdir)],
                cwd=str(PROJECT_ROOT),
                stdout=log_file, stderr=subprocess.STDOUT,
            )

        # poll for completion
        finished = [i for i, p in running.items() if p.poll() is not None]
        for i in finished:
            rc = running.pop(i).returncode
            if rc == 0 and (outdir / f"job_{i}.pkl").exists():
                done.add(i)
            else:
                failed.append(i)
                print(f"[schedule] job {i} ({jobs[i]['label']}) failed, exit code {rc}", flush=True)

        # write progress every ~30 seconds
        now = time.time()
        if now - last_progress_write > 30:
            last_progress_write = now
            running_labels = [f"{i}:{jobs[i]['label']}" for i in running]
            with open(progress, "w", encoding="utf-8") as pf:
                pf.write(f"[{time.strftime('%H:%M:%S')}] done {len(done)}/{len(jobs)} | "
                         f"failed {len(failed)} | running {len(running)}\n")
                if running_labels:
                    pf.write("running: " + ", ".join(running_labels) + "\n")
                if failed:
                    pf.write("failed: " + ", ".join(str(i) for i in failed) + "\n")

        if not finished:
            time.sleep(10)

    with open(progress, "a", encoding="utf-8") as pf:
        pf.write(f"\nAll finished: done {len(done)}/{len(jobs)}, failed {len(failed)}\n")
    return done, failed


def consolidate(jobs: List[dict], outdir: Path, done: set, failed: List[int]):
    """Rebuild output files identical to the serial scripts (CSV + PNG + summary table)."""
    from collections import defaultdict
    results: Dict[str, List[dict]] = defaultdict(list)
    curves: Dict[str, list] = {}

    for i, job in enumerate(jobs):
        pkl = outdir / f"job_{i}.pkl"
        if not pkl.exists():
            print(f"[rebuild] skipping missing job {i} ({job['label']})", flush=True)
            continue
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        results[job["group"]].append(data["stats"])
        if job.get("need_curve") and data.get("curve"):
            curves[job["label"]] = data["curve"]

    def save(group_name, filename):
        if results.get(group_name):
            C.save_csv(results[group_name], filename)

    def plot(group_name, filename, title):
        if results.get(group_name):
            C.plot_comparison(results[group_name], filename, title)

    # ---- Stage1 ----
    save("stage1_temp", "stage1_temperature_summary.csv")
    plot("stage1_temp", "stage1_temperature_comparison.png",
         "Stage1: Temperature ON vs OFF (Black-box Tactile Feedback)")
    save("stage1_sampling", "stage1_sampling_summary.csv")
    plot("stage1_sampling", "stage1_sampling_comparison.png",
         "Stage1: Micro-sampling Strategy Comparison (9 Methods)")
    if results.get("stage1_temp") or results.get("stage1_sampling"):
        C.save_csv(results["stage1_temp"] + results["stage1_sampling"],
                   "stage1_summary.csv")
        C.print_summary(results["stage1_temp"] + results["stage1_sampling"], "Stage1")

    # ---- Stage2 ----
    save("stage2_finder", "stage2_finder_summary.csv")
    plot("stage2_finder", "stage2_finder_comparison.png",
         "Stage2: Macro-traversal Strategy Comparison (5 Methods)")
    save("stage2_threshold", "stage2_threshold_summary.csv")
    plot("stage2_threshold", "stage2_threshold_comparison.png",
         "Stage2: Deep-sea Distance Threshold Sweep [416,448,480,512]")
    if results.get("stage2_finder") or results.get("stage2_threshold"):
        C.save_csv(results["stage2_finder"] + results["stage2_threshold"],
                   "stage2_summary.csv")
        C.print_summary(results["stage2_finder"] + results["stage2_threshold"], "Stage2")

    # ---- Stage3 ----
    save("stage3_reward", "stage3_summary.csv")
    plot("stage3_reward", "stage3_comparison.png",
         "Stage3: Reward Module Ablation (4 Configurations)")
    if results.get("stage3_reward"):
        C.print_summary(results["stage3_reward"], "Stage3")

    # ---- Stage4 ----
    save("stage4_curiosity", "stage4_curiosity_summary.csv")
    plot("stage4_curiosity", "stage4_curiosity_comparison.png",
         "Stage4: Curiosity Weight Sweep [1,3,5]")
    save("stage4_budget", "stage4_budget_summary.csv")
    plot("stage4_budget", "stage4_budget_comparison.png",
         "Stage4: Query Budget Sweep")
    if curves:
        C.save_convergence_csv(curves, "stage4_convergence.csv")
        C.plot_convergence_curves(curves, "stage4_convergence_curves.png",
                                  "Stage4: Convergence Curves (Query Count vs Deep Sea Points)")
    if results.get("stage4_curiosity") or results.get("stage4_budget"):
        C.save_csv(results["stage4_curiosity"] + results["stage4_budget"],
                   "stage4_summary.csv")
        C.print_summary(results["stage4_curiosity"] + results["stage4_budget"], "Stage4")

    if failed:
        print(f"\n[!] {len(failed)} jobs failed: "
              + ", ".join(f"job{i}({jobs[i]['label']})" for i in failed))
        print("  Check outdir/job_<i>.err and job_<i>.log to find the cause and rerun the corresponding job.")


def main():
    parser = argparse.ArgumentParser(description="Run all Stage1~4 ablation experiments in parallel")
    parser.add_argument("--workers", type=int, default=12,
                        help="number of concurrent subprocesses (default 12)")
    parser.add_argument("--python", default="",
                        help="Python interpreter command (default: the absolute path of the current interpreter)")
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "results" / "parallel_jobs"),
                        help="directory for intermediate artifacts/logs (default results/parallel_jobs/)")
    parser.add_argument("--stage", choices=["all", "1", "2", "3", "4"], default="all",
                        help="run only the given stage (default all)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    only_stage = None if args.stage == "all" else args.stage
    py_cmd = args.python or sys.executable   # default to the absolute path of the current interpreter to avoid PATH lookup failures

    jobs = build_jobs(only_stage)
    if not jobs:
        print("[fail] no jobs were built, please check configs/ablation/")
        sys.exit(1)

    print("=" * 74)
    print("  Parallel ablation experiment scheduler (runs Stage1~4 to completion)")
    print("=" * 74)
    print(f"  Total config jobs: {len(jobs)} (each = 20 seeds x 20 repeats = 400 runs)")
    print(f"  Concurrent subprocesses: {args.workers} | Python: {py_cmd}")
    print(f"  Output directory: {outdir}")
    print("=" * 74)
    for i, j in enumerate(jobs):
        print(f"    job {i:>2} [{j['group']:<16}] {j['label']}")

    # write the manifest for the worker to read
    with open(outdir / "manifest.pkl", "wb") as f:
        pickle.dump(jobs, f)

    done, failed = run_jobs(jobs, outdir, args.workers, py_cmd)
    consolidate(jobs, outdir, done, failed)

    print("\n[done] All parallel ablation experiments finished."
          "Result files are in results/ablation_logs/ (same as the serial scripts' output).")


if __name__ == "__main__":
    main()
