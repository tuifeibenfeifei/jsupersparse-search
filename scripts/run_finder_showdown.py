"""
run_finder_showdown.py
Stage2 精简对比：仅 ray_scan vs grid_scan（用户指定，避免全 5 策略耗时过长）。

用途：决定最终版采用哪个宏观遍历策略。
规模：20 世界种子 × 20 次重复 = 400 次运行 / 单策略（需求第十二节）。

⚠️ 支持 --finder 参数单独运行一个策略，以便两个策略并行执行、缩短墙钟时间：
       python run_finder_showdown.py --finder ray_scan
       python run_finder_showdown.py --finder grid_scan
   不带 --finder 时顺序运行两者并输出汇总对比表。

数值全部来自 configs/ablation/stage2_movement/finder_sweep.yaml（需求第十节：
配置驱动，严禁硬编码消融数值）。
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _ablation_common import (
    CONFIG_DIR, load_yaml, build_config, run_single_config,
    save_csv, plot_comparison, print_summary,
)

SHOWDOWN_YAML = "stage2_movement/finder_sweep.yaml"
CANDIDATE_FINDERS = ["ray_scan", "grid_scan"]


def run_one(finder: str, shared: dict, mode: str):
    """运行单个宏观策略，返回统计字典"""
    config = build_config(shared, {"chunk_finder": finder}, mode)
    return run_single_config(config, f"finder={finder}", "Stage2-精简")


def main():
    parser = argparse.ArgumentParser(description="Stage2 精简对比：ray_scan vs grid_scan")
    parser.add_argument("--mode", default="simple", choices=["simple", "detailed"],
                        help="输出模式：simple=每配置20行平均值；detailed=完整结果+实时查询")
    parser.add_argument("--finder", default=None, choices=CANDIDATE_FINDERS,
                        help="仅运行指定策略（用于并行执行）；缺省时顺序运行两者")
    parser.add_argument("--no-plot", action="store_true", help="只输出 CSV，不生成 PNG")
    args = parser.parse_args()

    yaml_cfg = load_yaml(CONFIG_DIR / SHOWDOWN_YAML)
    # 列表键为调度元信息，不可直接传入 Explorer（需求第九节）
    shared = {k: v for k, v in yaml_cfg.items() if k != "chunk_finder_list"}

    finders = [args.finder] if args.finder else CANDIDATE_FINDERS

    if args.finder:
        # 单策略模式：供并行调用，各自写独立 CSV 避免竞争
        stats = run_one(args.finder, shared, args.mode)
        save_csv([stats], f"stage2_{args.finder}_summary.csv")
        print_summary([stats], f"Stage2 单策略结果: {args.finder}")
        return

    print("=" * 74)
    print("  Stage2 精简对比：ray_scan vs grid_scan")
    print("=" * 74)
    print(f"  输出模式    : {'简易版 (20行平均值)' if args.mode == 'simple' else '详细版 (完整结果+实时查询输出)'}")
    print(f"  候选策略    : {', '.join(finders)}")
    print(f"  继承基准    : micro={shared.get('micro_strategy')}, "
          f"threshold={shared.get('deep_sea_distance_threshold')}")
    print(f"  温度架构    : in_state={shared.get('temperature_in_state')}, "
          f"guidance={shared.get('temperature_guidance_enabled')}")
    print(f"  查询上限    : {shared.get('max_queries_per_episode')} / "
          f"目标 {shared.get('target_deep_sea_points')} 点")

    results = [run_one(f, shared, args.mode) for f in finders]

    csv_path = save_csv(results, "stage2_finder_showdown_summary.csv")
    if not args.no_plot:
        plot_comparison(results, "stage2_finder_showdown_comparison.png",
                        "Stage2 Finder Showdown (ray_scan vs grid_scan)")
    print_summary(results, "Stage2 精简对比 (ray_scan vs grid_scan)")

    # 最终版选型建议
    best = max(results, key=lambda s: s.get("mean_points", 0))
    print(f"\n  ▶ 最终版建议采用: {best['label']}  "
          f"(平均深海点 {best.get('mean_points', 0):.2f})")
    if csv_path:
        print(f"  ▶ CSV 已保存: {csv_path}")


if __name__ == "__main__":
    main()
