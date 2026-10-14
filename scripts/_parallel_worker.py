"""
_parallel_worker.py
并行消融实验的单配置执行子进程（由 run_stages_parallel.py 调度，勿单独运行）。

【职责】
    接收一个 job（配置字典 + 标签）→ 调用 DeepSeaExplorer.run_experiment()
    跑 20 种子 × 20 次 → 将统计结果（剔除 raw_results 以控制体积）与 Stage4
    需要的收敛曲线数据以 pickle 写入 outdir/job_<idx>.pkl。

【退出码】
    0 = 成功（job_<idx>.pkl 已生成）
    1 = 失败（错误堆栈写入 job_<idx>.err，不影响其他 job）
"""

import sys
import pickle
import argparse
import traceback
from pathlib import Path

# 项目根目录 + scripts 目录入路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description="并行消融单配置 worker")
    parser.add_argument("--manifest", required=True, help="job 清单(pickle)路径")
    parser.add_argument("--job-idx", type=int, required=True, help="job 索引")
    parser.add_argument("--outdir", required=True, help="输出目录")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest, "rb") as f:
        jobs = pickle.load(f)
    job = jobs[args.job_idx]
    label = job["label"]

    try:
        print(f"[worker {args.job_idx}] {label} 启动", flush=True)

        import _ablation_common as C
        from src.main_algorithm import DeepSeaExplorer

        config = dict(job["config"])
        config["output_mode"] = "simple"   # 并行跑统一用简易版（每种子一行平均值）

        explorer = DeepSeaExplorer(config)
        stats = explorer.run_experiment(
            world_seeds=C.WORLD_SEEDS,
            repeats=C.REPEATS,
            algo_seed_start=42,
        )
        stats["label"] = label

        # Stage4 预算扫描需要收敛曲线（与原脚本 run_query_target_sweep 同一聚合逻辑）
        curve = None
        if job.get("need_curve"):
            import scripts.run_ablation_stage4 as s4
            curve = s4._aggregate_convergence(stats.get("raw_results", []))

        # 剔除 raw_results（体积过大），聚合统计足够重建全部输出文件
        stats.pop("raw_results", None)

        with open(outdir / f"job_{args.job_idx}.pkl", "wb") as f:
            pickle.dump({"stats": stats, "curve": curve}, f)

        print(f"[worker {args.job_idx}] {label} 完成 | "
              f"平均深海点={stats.get('mean_points', 0):.2f} | "
              f"平均查询={stats.get('mean_queries', 0):.1f}", flush=True)

    except Exception:
        err_path = outdir / f"job_{args.job_idx}.err"
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print(f"[worker {args.job_idx}] {label} 失败，详见 {err_path.name}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
