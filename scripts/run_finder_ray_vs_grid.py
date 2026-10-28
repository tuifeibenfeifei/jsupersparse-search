"""
run_finder_ray_vs_grid.py
Stage2 focused comparison: ray_scan vs grid_scan (full 20 seeds x 20 repeats = 400 runs each).

[Responsibility]
    User requirement (final version): only ray_scan and grid_scan need to be compared in Stage2;
    the final model adopts whichever of the two wins on query efficiency.
    This script runs that head-to-head with the exact same locked settings that every other
    stage inherits (Stage1 optimum + all rewards on + 40 points / 10000 query cap), so the
    winner can be copied straight into configs/final_model.yaml.

[Parallelism]
    One subprocess per world seed (20 seeds -> 20 processes), each running 20 repeats,
    then results are merged into a single CSV + bar chart.

[Command line]
    --workers N   concurrent processes (default 10)
    --repeats N   repeats per world seed (default 20)
    --mode        simple / detailed (default simple)

[Output]
    results/ablation_logs/stage2_ray_vs_grid_summary.csv
    results/ablation_logs/stage2_ray_vs_grid_comparison.png
"""

import sys
import csv
import json
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import _ablation_common as C

FINDERS = ["ray_scan", "grid_scan"]
BASE_YAML = C.PROJECT_ROOT / "configs" / "ablation" / "stage2_movement" / "finder_sweep.yaml"
OUT_DIR = C.PROJECT_ROOT / "results" / "ablation_logs"

WORKER_CODE = r"""
import sys, json
sys.path.insert(0, r"{root}")
sys.path.insert(0, r"{scripts}")
from src.main_algorithm import DeepSeaExplorer
import _ablation_common as C

cfg = dict(C.load_yaml(r"{yaml}"))
cfg.pop("chunk_finder_list", None)
cfg["chunk_finder"] = "{finder}"
cfg["output_mode"] = "{mode}"

st = DeepSeaExplorer(cfg).run_experiment(world_seeds=[{seed}], repeats={repeats}, algo_seed_start=42)
out = {{
    "finder": "{finder}",
    "seed": {seed},
    "mean_points": st.get("mean_points", 0.0),
    "std_points": st.get("std_points", 0.0),
    "mean_queries": st.get("mean_queries", 0.0),
    "std_queries": st.get("std_queries", 0.0),
    "success_rate": st.get("success_rate", 0.0),
    "target_reached_count": st.get("target_reached_count", 0),
    "n_runs": st.get("n_runs", 0),
}}
print("RESULT_JSON " + json.dumps(out))
"""


def aggregate(jobs: List[Dict], finder: str) -> Dict[str, float]:
    """Merge per-seed stats into one config-level row (weighted by n_runs)."""
    sel = [j for j in jobs if j["finder"] == finder]
    if not sel:
        return {}
    n = sum(j.get("n_runs", 0) for j in sel) or len(sel)
    mean_points = sum(j["mean_points"] * j.get("n_runs", 1) for j in sel) / n
    mean_queries = sum(j["mean_queries"] * j.get("n_runs", 1) for j in sel) / n
    sq = sum((j["std_queries"] ** 2 + j["mean_queries"] ** 2) * j.get("n_runs", 1) for j in sel) / n
    std_queries = max(sq - mean_queries ** 2, 0.0) ** 0.5
    sqp = sum((j["std_points"] ** 2 + j["mean_points"] ** 2) * j.get("n_runs", 1) for j in sel) / n
    std_points = max(sqp - mean_points ** 2, 0.0) ** 0.5
    reached = sum(j.get("target_reached_count", 0) for j in sel)
    return {
        "label": f"finder={finder}",
        "mean_points": mean_points,
        "std_points": std_points,
        "mean_queries": mean_queries,
        "std_queries": std_queries,
        "success_rate": mean_points / 40.0 if mean_points else 0.0,
        "target_reached_count": reached,
        "n_runs": n,
    }


def main():
    C.require_real_cubiomes()   # refuse to run on Mock (missing cubiomes.dll)
    parser = argparse.ArgumentParser(description="Stage2 ray vs grid full comparison")
    parser.add_argument("--workers", type=int, default=10, help="concurrent processes")
    parser.add_argument("--repeats", type=int, default=20, help="repeats per world seed")
    parser.add_argument("--mode", choices=["simple", "detailed"], default="simple")
    args = parser.parse_args()

    C.print_stage_banner(
        "Stage2 focused comparison (ray_scan vs grid_scan)", args.mode,
        f"{len(FINDERS)} macro-traversal strategies x {len(C.WORLD_SEEDS)} seeds "
        f"x {args.repeats} repeats = {len(FINDERS) * C.TOTAL_RUNS} runs",
    )

    workdir = Path(tempfile.mkdtemp(prefix="ray_vs_grid_"))
    queue = [(f, s) for f in FINDERS for s in C.WORLD_SEEDS]
    running = []
    jobs: List[Dict] = []

    while queue or running:
        while queue and len(running) < args.workers:
            finder, seed = queue.pop(0)
            p = subprocess.Popen(
                [sys.executable, "-c", WORKER_CODE.format(
                    root=str(PROJECT_ROOT), scripts=str(Path(__file__).resolve().parent),
                    finder=finder, seed=seed, repeats=args.repeats, mode=args.mode,
                    yaml=str(BASE_YAML))],
                cwd=str(workdir), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            running.append((p, finder, seed))
        still = []
        for p, finder, seed in running:
            if p.poll() is None:
                still.append((p, finder, seed))
                continue
            out, err = p.communicate()
            payload = None
            for line in (out or "").splitlines():
                if line.startswith("RESULT_JSON "):
                    payload = json.loads(line[len("RESULT_JSON "):])
            if payload is None:
                print(f"[fail] finder={finder} seed={seed}\n{(err or '')[-1200:]}")
            else:
                jobs.append(payload)
                print(f"  done finder={finder:<10} seed={seed:<3} "
                      f"points={payload['mean_points']:.2f} queries={payload['mean_queries']:.1f}")
        running = still

    rows = [r for r in (aggregate(jobs, f) for f in FINDERS) if r]
    if not rows:
        print("[fail] no valid result")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "stage2_ray_vs_grid_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=C.CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in C.CSV_FIELDS})

    C.plot_comparison(rows, "stage2_ray_vs_grid_comparison.png",
                      "Stage2: ray_scan vs grid_scan (full 20x20)",
                      out_dir=OUT_DIR)
    C.print_summary(rows, "Stage2 ray vs grid",
                    next_hint="Copy the winner's chunk_finder value into configs/final_model.yaml")


if __name__ == "__main__":
    main()
