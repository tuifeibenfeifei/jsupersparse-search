"""
run_final_model.py
最终版模型运行脚本（L5 最终验证层）

【职责】对应 需求.txt 第十四节 run_final_model.py：
    读取 configs/final_model.yaml（Stage1~4 最优参数组合）→ 调用
    main_algorithm.run_experiment() 运行指定种子 × 重复次数 → 收集结果
    → 写入 results/final_logs/ → 绘制最终版分析图表。

【输出双版本】（与各 Stage 脚本同一约定）：
    simple   -> 简易版：每个世界种子输出一行平均值（N 行，N=种子数）
    detailed -> 详细版：输出每次运行的完整结果 + 实时查询输出

【命令行】
    --mode    simple / detailed（默认 simple，便于批量出数）
    --seeds   种子范围，如 1-20 或 1,2,3（默认 1-20）
    --repeats 每个种子重复次数（默认 20）
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ablation_common as C

FINAL_MODEL_YAML = C.PROJECT_ROOT / "configs" / "final_model.yaml"
FINAL_RESULTS_DIR = C.PROJECT_ROOT / "results" / "final_logs"


def parse_seed_range(spec: str):
    """解析种子范围：'1-20' -> [1..20]；'1,3,5' -> [1,3,5]"""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s.strip()]


def main():
    parser = argparse.ArgumentParser(description="最终版模型运行脚本（L5 最终验证层）")
    parser.add_argument("--mode", choices=["simple", "detailed"], default="simple",
                        help="输出模式：simple=每种子一行平均值；detailed=每次运行完整结果+实时查询")
    parser.add_argument("--seeds", default="1-20",
                        help="世界种子范围，如 1-20 或 1,2,3（默认 1-20）")
    parser.add_argument("--repeats", type=int, default=20,
                        help="每个世界种子的重复次数（默认 20）")
    args = parser.parse_args()

    if not FINAL_MODEL_YAML.exists():
        print(f"[失败] 配置文件不存在: {FINAL_MODEL_YAML}")
        print("请先依据 Stage1~4 消融结果填写 configs/final_model.yaml")
        sys.exit(1)

    world_seeds = parse_seed_range(args.seeds)
    config = C.load_yaml(FINAL_MODEL_YAML)
    config["output_mode"] = args.mode

    C.print_stage_banner(
        "L5 最终版模型验证", args.mode,
        f"种子 {world_seeds[0]}~{world_seeds[-1]} × {args.repeats} 次重复，"
        f"配置来源 configs/final_model.yaml",
    )
    FINAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    from src.main_algorithm import DeepSeaExplorer

    explorer = DeepSeaExplorer(config)
    stats = explorer.run_experiment(
        world_seeds=world_seeds,
        repeats=args.repeats,
        algo_seed_start=42,
    )

    # 写入最终模型统计 CSV + 收敛曲线 CSV/PNG（需求第十四节：results/final_logs/）
    stats["label"] = "final_model"
    C.save_csv([stats], "final_model_summary.csv", out_dir=FINAL_RESULTS_DIR)

    raw = stats.get("raw_results", [])
    if raw:
        # 按配置 label 打包收敛轨迹（所有运行合为一条平均曲线）
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

    C.print_summary([stats], "L5 最终版模型",
                    next_hint="最终版运行完毕：结果见 results/final_logs/")

    # 打印最终版配置要点（需求第十节：动态输出说明头在 DeepSeaExplorer 内已打印）
    print("\n  ▶ 最终版配置摘要")
    print(f"    macro={config.get('chunk_finder')} | micro={config.get('micro_strategy')} "
          f"| threshold={config.get('deep_sea_distance_threshold')}")
    print(f"    temp={config.get('temperature_enabled')} | "
          f"guidance={config.get('temperature_guidance_enabled')} | "
          f"curiosity_weight={config.get('curiosity_weight')}")
    print(f"    budget={config.get('max_queries_per_episode')} queries / "
          f"target {config.get('target_deep_sea_points')} points")


if __name__ == "__main__":
    main()
