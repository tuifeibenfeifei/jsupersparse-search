"""
run_stages_parallel.py
并行完整跑完 Stage1~4 全部消融实验（等价于依次运行 run_ablation_stage1~4.py，
但以"一个配置 = 一个子进程"的方式并行调度，大幅缩短总耗时）。

【设计】
    · 31 个配置 job（Stage1: 2温度 + 9策略 / Stage2: 5遍历 + 4阈值 /
      Stage3: 4奖励 / Stage4: 3权重 + 4预算），每个 job = 20 种子 × 20 次 = 400 次运行
    · job 构建逻辑与各 stage 脚本完全一致（同一 YAML、同一 label、同一注入键）
    · 以 N 个并发子进程执行（--workers，默认 12），单个 job 崩溃不影响其他 job
    · 全部完成后，按各 stage 脚本的命名与绘图逻辑重建输出文件
      （stageX_*.csv / stageX_*.png / stageX_summary.csv），与串行跑完全等价

【命令行】
    --workers N    并发子进程数（默认 12；机器 16 线程，建议 8~14）
    --python CMD   Python 解释器命令（默认 "py -3.9"）
    --outdir DIR   中间产物与日志目录（默认 results/parallel_jobs/）
    --stage all|1|2|3|4   只跑指定 stage（默认 all）

【进度查看】
    outdir/progress.txt —— 实时进度（已完成/总数、运行中 job、失败 job）
    outdir/job_<i>.log   —— 每个 job 的运行日志
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

# Stage3 的 4 个奖励配置（与 run_ablation_stage3.py 的 REWARD_CONFIGS 一致）
STAGE3_REWARD_CONFIGS = [
    ("reward_all_on", "all_on(全开对照)"),
    ("reward_no_deep", "no_deep(关深海)"),
    ("reward_no_curiosity", "no_curiosity(关好奇)"),
    ("reward_no_penalty", "no_penalty(关惩罚)"),
]

# 各 stage 输出文件重建信息：分组 -> (输出文件名列表, 绘图信息)
# 绘图信息: (png 文件名, 标题)


def build_jobs(only_stage: Optional[str] = None) -> List[dict]:
    """构建全部 job（与各 stage 脚本完全相同的配置构造逻辑）"""
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

    # ---------------- Stage1 感知层 ----------------
    if only_stage in (None, "1"):
        # 温度 on/off（pop 掉扫描列表键，与原脚本一致）
        for name in ["temp_on", "temp_off"]:
            add("stage1_temp", f"stage1_perception/{name}.yaml",
                {}, name, pop_fields=("micro_strategy_list",))
        # 9 种微观策略扫描
        sweep = C.load_yaml(ABLATION_DIR / "stage1_perception/sampling_sweep.yaml")
        shared = {k: v for k, v in sweep.items() if k != "micro_strategy_list"}
        for s in sweep.get("micro_strategy_list", []):
            jobs.append({"group": "stage1_sampling",
                         "config": {**shared, "micro_strategy": s},
                         "label": s, "need_curve": False})

    # ---------------- Stage2 移动层 ----------------
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

    # ---------------- Stage3 奖励层 ----------------
    if only_stage in (None, "3"):
        for cfg_name, label in STAGE3_REWARD_CONFIGS:
            add("stage3_reward", f"stage3_reward/{cfg_name}.yaml", {}, label)

    # ---------------- Stage4 超参数层 ----------------
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
    """以 workers 个并发子进程执行全部 job，返回 (完成索引集合, 失败索引列表)"""
    pending = list(range(len(jobs)))
    running: Dict[int, subprocess.Popen] = {}
    done: set = set()
    failed: List[int] = []
    progress = outdir / "progress.txt"
    last_progress_write = 0.0

    with open(progress, "w", encoding="utf-8") as pf:
        pf.write(f"共 {len(jobs)} 个配置 job，并发 {workers} 个\n")

    while pending or running:
        # 启动新 job（水位控制；已有结果的 job 直接跳过 = 续跑/断点恢复）
        while len(running) < workers and pending:
            idx = pending.pop(0)
            if (outdir / f"job_{idx}.pkl").exists():
                done.add(idx)
                print(f"[调度] job {idx} ({jobs[idx]['label']}) 已有结果，跳过", flush=True)
                continue
            log_path = outdir / f"job_{idx}.log"
            log_file = open(log_path, "a", encoding="utf-8")
            print(f"[调度] 启动 job {idx} ({jobs[idx]['label']})", flush=True)
            running[idx] = subprocess.Popen(
                [py_cmd, str(WORKER), "--manifest", str(outdir / "manifest.pkl"),
                 "--job-idx", str(idx), "--outdir", str(outdir)],
                cwd=str(PROJECT_ROOT),
                stdout=log_file, stderr=subprocess.STDOUT,
            )

        # 轮询完成情况
        finished = [i for i, p in running.items() if p.poll() is not None]
        for i in finished:
            rc = running.pop(i).returncode
            if rc == 0 and (outdir / f"job_{i}.pkl").exists():
                done.add(i)
            else:
                failed.append(i)
                print(f"[调度] job {i} ({jobs[i]['label']}) 失败，退出码 {rc}", flush=True)

        # 每 ~30 秒写一次进度
        now = time.time()
        if now - last_progress_write > 30:
            last_progress_write = now
            running_labels = [f"{i}:{jobs[i]['label']}" for i in running]
            with open(progress, "w", encoding="utf-8") as pf:
                pf.write(f"[{time.strftime('%H:%M:%S')}] 完成 {len(done)}/{len(jobs)} | "
                         f"失败 {len(failed)} | 运行中 {len(running)} 个\n")
                if running_labels:
                    pf.write("运行中: " + ", ".join(running_labels) + "\n")
                if failed:
                    pf.write("失败: " + ", ".join(str(i) for i in failed) + "\n")

        if not finished:
            time.sleep(10)

    with open(progress, "a", encoding="utf-8") as pf:
        pf.write(f"\n全部结束：完成 {len(done)}/{len(jobs)}，失败 {len(failed)}\n")
    return done, failed


def consolidate(jobs: List[dict], outdir: Path, done: set, failed: List[int]):
    """重建与串行脚本完全一致的输出文件（CSV + PNG + 汇总表）"""
    from collections import defaultdict
    results: Dict[str, List[dict]] = defaultdict(list)
    curves: Dict[str, list] = {}

    for i, job in enumerate(jobs):
        pkl = outdir / f"job_{i}.pkl"
        if not pkl.exists():
            print(f"[重建] 跳过缺失 job {i} ({job['label']})", flush=True)
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
        print(f"\n⚠️ 有 {len(failed)} 个 job 失败: "
              + ", ".join(f"job{i}({jobs[i]['label']})" for i in failed))
        print("  可查看 outdir/job_<i>.err 与 job_<i>.log 定位原因，重跑对应 job。")


def main():
    parser = argparse.ArgumentParser(description="并行完整跑完 Stage1~4 消融实验")
    parser.add_argument("--workers", type=int, default=12,
                        help="并发子进程数（默认 12）")
    parser.add_argument("--python", default="",
                        help="Python 解释器命令（默认使用当前解释器绝对路径）")
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "results" / "parallel_jobs"),
                        help="中间产物/日志目录（默认 results/parallel_jobs/）")
    parser.add_argument("--stage", choices=["all", "1", "2", "3", "4"], default="all",
                        help="只跑指定 stage（默认 all）")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    only_stage = None if args.stage == "all" else args.stage
    py_cmd = args.python or sys.executable   # 默认用当前解释器绝对路径，避免 PATH 解析失败

    jobs = build_jobs(only_stage)
    if not jobs:
        print("[失败] 未构建出任何 job，请检查 configs/ablation/")
        sys.exit(1)

    print("=" * 74)
    print("  并行消融实验调度器（Stage1~4 完整跑完）")
    print("=" * 74)
    print(f"  配置 job 总数: {len(jobs)}（每个 = 20 种子 × 20 次 = 400 次运行）")
    print(f"  并发子进程 : {args.workers} | Python: {py_cmd}")
    print(f"  输出目录   : {outdir}")
    print("=" * 74)
    for i, j in enumerate(jobs):
        print(f"    job {i:>2} [{j['group']:<16}] {j['label']}")

    # 写 manifest 供 worker 读取
    with open(outdir / "manifest.pkl", "wb") as f:
        pickle.dump(jobs, f)

    done, failed = run_jobs(jobs, outdir, args.workers, py_cmd)
    consolidate(jobs, outdir, done, failed)

    print("\n[完成] 并行消融实验全部结束。"
          "结果文件位于 results/ablation_logs/（与串行脚本输出一致）。")


if __name__ == "__main__":
    main()
