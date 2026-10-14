"""
run_baselines.py
5 个基线算法对比实验调度脚本（L4 基线对比层）

【职责】对应 需求.txt 第十四节 run_baselines.py：
    读取 configs/ablation/baselines/ 下的 5 个 YAML → 循环调用各基线算法的
    run_xxx(config, seed) → 收集结果 → 聚合统计 → 写入 results/baseline_logs/
    → 绘制对比图。

【基线算法】src/baselines/ 下的 5 个独立算法（只依赖 L0）：
    bfs / go_explore / icm / random_sampling / rnd
    每个算法的 run 函数返回与 main_algorithm.run_single_episode() 同构的结果字典。

【随机性分离】需求第十二节：
    世界种子控制地形（传入 run_xxx 的 seed 参数），
    算法随机性由 config["algo_seed"] 逐次递增注入，保证同一世界种子
    的多次重复结果不同（分离环境随机性与算法随机性）。

【命令行】--algo 选择子集 / --seeds 种子范围 / --repeats 重复次数
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ablation_common as C

BASELINE_CONFIG_DIR = C.PROJECT_ROOT / "configs" / "ablation" / "baselines"
BASELINE_RESULTS_DIR = C.PROJECT_ROOT / "results" / "baseline_logs"

# 基线算法注册表：YAML 文件名（不含后缀） -> 运行函数
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
    """延迟导入 5 个基线算法（避免脚本 import 时加载 numpy 网络）"""
    for name, (module_name, func_name) in _BASELINE_MODULES.items():
        module = __import__(module_name, fromlist=[func_name])
        BASELINES[name] = getattr(module, func_name)


def run_single_baseline(algo_name: str, config: dict, world_seed: int,
                        algo_seed: int) -> dict:
    """
    运行一次基线 episode。
    - config 注入 algo_seed（算法随机种子），世界种子作为 run_xxx 的 seed 参数。
    - 返回结果字典（与 run_single_episode 同构）。
    """
    cfg = dict(config)
    cfg["algo_seed"] = algo_seed
    return BASELINES[algo_name](cfg, world_seed)


def run_baseline_config(algo_name: str, config: dict, world_seeds: List[int],
                        repeats: int, algo_seed_start: int = 42) -> Dict[str, float]:
    """
    对单个基线算法执行 world_seeds × repeats 次实验并聚合统计。
    Returns: 与 run_experiment 同构的统计字典（含 label）。
    """
    all_points: List[int] = []
    all_queries: List[int] = []
    raw_results: List[dict] = []
    target = int(config.get("target_deep_sea_points", 40))
    algo_seed_counter = algo_seed_start

    print(f"\n{'=' * 74}")
    print(f"▶ 基线算法: {algo_name}")
    print(f"  世界种子 {world_seeds[0]}~{world_seeds[-1]} × 每种子重复 {repeats} 次 "
          f"= {len(world_seeds) * repeats} 次运行")
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
                print(f"\n  [异常] {algo_name} world_seed={world_seed} "
                      f"algo_seed={algo_seed} 失败: {e}", file=sys.stderr)
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
        print(f"  [简易版] 世界种子 {world_seed:>2} | 平均深海点={avg_p:>6.2f} | "
              f"平均查询={avg_q:>7.1f} | "
              f"成功率={avg_p:.2f}/{target} ({seed_rate * 100:>5.1f}%)")

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

    print(f"  [总体] 平均深海点={mean_points:.2f}±{std_points:.2f} | "
          f"平均查询={mean_queries:.1f}±{std_queries:.1f} | "
          f"成功率={mean_points:.2f}/{target} ({success_rate * 100:.1f}%) | "
          f"跑满目标次数={target_reached_count}/{n}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="5 个基线算法对比实验调度脚本")
    parser.add_argument("--algo", default="all",
                        choices=["all"] + sorted(_BASELINE_MODULES.keys()),
                        help="运行范围：all=全部 5 个基线，或指定单个")
    parser.add_argument("--seeds", default="1-20",
                        help="世界种子范围，如 1-20 或 1,2,3（默认 1-20）")
    parser.add_argument("--repeats", type=int, default=20,
                        help="每个世界种子的重复次数（默认 20）")
    args = parser.parse_args()

    world_seeds = parse_seed_range(args.seeds)
    C.print_stage_banner(
        "L4 基线算法对比实验", "simple",
        f"BFS / Go-Explore / ICM / Random Sampling / RND "
        f"（种子 {world_seeds[0]}~{world_seeds[-1]} × {args.repeats} 次重复）",
    )
    BASELINE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _load_baseline_functions()

    algo_names = (sorted(_BASELINE_MODULES.keys())
                  if args.algo == "all" else [args.algo])

    all_stats: List[Dict[str, float]] = []
    for algo_name in algo_names:
        yaml_path = BASELINE_CONFIG_DIR / f"{algo_name}.yaml"
        if not yaml_path.exists():
            print(f"[跳过] 配置文件不存在: {yaml_path}")
            continue
        config = C.load_yaml(yaml_path)
        stats = run_baseline_config(algo_name, config, world_seeds, args.repeats)
        all_stats.append(stats)

    if not all_stats:
        print("\n[失败] 未找到任何有效基线配置，请检查 configs/ablation/baselines/")
        sys.exit(1)

    C.save_csv(all_stats, "baselines_summary.csv", out_dir=BASELINE_RESULTS_DIR)
    C.plot_comparison(all_stats, "baselines_comparison.png",
                      "Baselines Comparison (BFS / Go-Explore / ICM / Random / RND)",
                      out_dir=BASELINE_RESULTS_DIR)
    C.print_summary(all_stats, "L4 基线算法对比",
                    next_hint="下一步：将最终版框架与上述 5 个基线横向对比，"
                              "证明黑盒学习框架的样本效率优势")


def parse_seed_range(spec: str) -> List[int]:
    """解析种子范围：'1-20' -> [1..20]；'1,3,5' -> [1,3,5]"""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s.strip()]


if __name__ == "__main__":
    main()
