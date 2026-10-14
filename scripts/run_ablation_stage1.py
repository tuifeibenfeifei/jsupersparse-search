"""
run_ablation_stage1.py
Stage1 感知层消融实验调度脚本

【职责】严格对应 需求.txt 第十四节 run_ablation_stage1.py 条目：
    数据流向：读取 stage1_perception/ 下的 YAML → 循环调用 main_algorithm 跑
              20种子 × 20次 → 生成 CSV 和图表 → 存入 results/ablation_logs/
    调用关系：调用 main_algorithm.run_experiment()，调用 matplotlib 画图
    得到结果：stage1_summary.csv 和 stage1_comparison.png（确定最优感知方式）

【需求第十节】输出双版本：
    简易版 (output_mode=simple)   -> 每个配置输出 20 行平均值
    详细版 (output_mode=detailed) -> 输出 20 次完整结果 + 实时查询输出

【消融变量】需求第八节 Stage1（基础感知层）：
    固定其他所有变量为默认值，仅测试【温度 on/off】与【微观选点 9 种】。
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import matplotlib
matplotlib.use("Agg")   # 无窗口环境下也能出图
import matplotlib.pyplot as plt

# 将项目根目录加入路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.main_algorithm import DeepSeaExplorer

# ==================== 实验常量（需求第七节 / 第十二节） ====================
WORLD_SEEDS = list(range(1, 21))    # 所有实验在种子 1~20 上运行（需求第七节）
REPEATS = 20                        # 同一种子重复 20 次取平均（需求第十二节）
TOTAL_RUNS = len(WORLD_SEEDS) * REPEATS  # 单配置总运行次数（400次）

STAGE1_CONFIG_DIR = PROJECT_ROOT / "configs" / "ablation" / "stage1_perception"
RESULTS_DIR = PROJECT_ROOT / "results" / "ablation_logs"

# 简易版/详细版开关，可由命令行 --mode 覆盖
OUTPUT_MODE = "simple"


def load_yaml(path: Path) -> dict:
    """安全加载YAML配置"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_config(base_config: dict, overrides: dict, mode: str) -> dict:
    """
    构造单配置：以 YAML 内容为唯一来源（需求第十节：配置文件驱动，严禁硬编码），
    仅注入输出模式。不覆盖任何 YAML 已声明的消融变量。
    """
    config = dict(base_config)
    config["output_mode"] = mode
    for key, value in overrides.items():
        config[key] = value
    return config


def run_single_config(config: dict, label: str) -> Dict[str, float]:
    """
    对单个配置执行 20世界种子 × 20次重复实验
    Returns: 统计字典 {label, mean_points, std_points, mean_queries, std_queries, n_runs, ...}
    """
    print(f"\n{'=' * 74}")
    print(f"▶ Stage1 消融配置: {label}")
    print(f"  世界种子 {WORLD_SEEDS[0]}~{WORLD_SEEDS[-1]} × 每种子重复 {REPEATS} 次 "
          f"= {TOTAL_RUNS} 次运行")
    print(f"{'=' * 74}")

    explorer = DeepSeaExplorer(config)

    stats = explorer.run_experiment(
        world_seeds=WORLD_SEEDS,
        repeats=REPEATS,
        algo_seed_start=42,
    )
    stats["label"] = label
    return stats


def run_temperature_ablation(mode: str) -> List[Dict[str, float]]:
    """
    运行温度 on/off 对照实验
    需求第八节 Stage1：温度感知 ON 作为黑盒创新基准，OFF 作为纯盲搜对照
    """
    results = []
    for name in ["temp_on", "temp_off"]:
        config_path = STAGE1_CONFIG_DIR / f"{name}.yaml"
        if not config_path.exists():
            print(f"[跳过] 配置文件不存在: {config_path}")
            continue
        yaml_config = load_yaml(config_path)
        # micro_strategy_list 是扫描列表字段，单配置实验中不传给 DeepSeaExplorer
        yaml_config.pop("micro_strategy_list", None)
        config = build_config(yaml_config, {}, mode)
        results.append(run_single_config(config, label=name))
    return results


def run_sampling_sweep(mode: str) -> List[Dict[str, float]]:
    """
    运行9种微观选点策略扫描
    需求第九节：sampling_sweep.yaml 包含 9 种微观策略列表，由本脚本遍历
    """
    sweep_path = STAGE1_CONFIG_DIR / "sampling_sweep.yaml"
    if not sweep_path.exists():
        print(f"[跳过] 配置文件不存在: {sweep_path}")
        return []

    sweep_config = load_yaml(sweep_path)
    strategy_list = sweep_config.get("micro_strategy_list", [])
    if not strategy_list:
        print("[跳过] sampling_sweep.yaml 中未找到 micro_strategy_list 列表")
        return []

    # 排除列表键，其余为所有策略共享的锁定基准值（需求第八节：非测试变量严格锁定）
    shared_config = {k: v for k, v in sweep_config.items() if k != "micro_strategy_list"}

    results = []
    for strategy in strategy_list:
        overrides = {"micro_strategy": strategy}
        config = build_config(shared_config, overrides, mode)
        results.append(run_single_config(config, label=strategy))
    return results


def save_csv(results: List[Dict[str, float]], filename: str) -> Optional[Path]:
    """保存统计结果到CSV（需求第十四节：存入 results/ablation_logs/）"""
    if not results:
        return None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RESULTS_DIR / filename
    # success_rate 口径 = 平均深海点数 / 目标深海点数（完成度）
    # target_reached_count = 真正把目标跑满的次数（辅助参考）
    fieldnames = ["label", "mean_points", "std_points", "mean_queries",
                  "std_queries", "success_rate", "target_reached_count", "n_runs"]

    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fieldnames})

    print(f"[保存] CSV 已写入: {filepath}")
    return filepath


def plot_comparison(results: List[Dict[str, float]], filename: str,
                    title: str, ylabel: str) -> Optional[Path]:
    """生成带误差棒的对比柱状图（需求第十二节：配合误差棒证明结果非偶然）"""
    if not results:
        return None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RESULTS_DIR / filename

    labels = [r["label"] for r in results]
    means = [r["mean_points"] for r in results]
    stds = [r["std_points"] for r in results]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 6))
    x_pos = list(range(len(labels)))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color="#2196F3", edgecolor="black")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.3,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"[保存] 图表已写入: {filepath}")
    return filepath


def parse_args():
    parser = argparse.ArgumentParser(description="Stage1 感知层消融实验调度脚本")
    parser.add_argument("--mode", choices=["simple", "detailed"], default=OUTPUT_MODE,
                        help="输出模式：simple=简易版(20行平均值)，detailed=详细版(20次完整结果+实时查询输出)")
    parser.add_argument("--part", choices=["all", "temperature", "sampling"], default="all",
                        help="运行范围：all=全部，temperature=仅温度on/off，sampling=仅9种微观策略扫描")
    return parser.parse_args()


def main():
    args = parse_args()
    mode = args.mode

    print("=" * 74)
    print("  Stage1 感知层消融实验")
    print("=" * 74)
    print(f"  实验规模    : 世界种子 {WORLD_SEEDS[0]}~{WORLD_SEEDS[-1]} "
          f"× 每种子重复 {REPEATS} 次 = 单配置 {TOTAL_RUNS} 次运行")
    print(f"  输出模式    : {'详细版 (20次完整结果 + 实时查询输出)' if mode == 'detailed' else '简易版 (20行平均值)'}")
    print(f"  消融变量    : 温度 on/off + 微观选点 9 种（其余变量锁定为 Stage1 基准值）")
    print(f"  输出目录    : {RESULTS_DIR}")
    print("=" * 74)

    temp_results: List[Dict[str, float]] = []
    sampling_results: List[Dict[str, float]] = []

    # 1. 温度 on/off 对照
    if args.part in ("all", "temperature"):
        temp_results = run_temperature_ablation(mode)
        if temp_results:
            save_csv(temp_results, "stage1_temperature_summary.csv")
            plot_comparison(
                temp_results,
                "stage1_temperature_comparison.png",
                "Stage1: Temperature ON vs OFF (Black-box Tactile Feedback)",
                "Average Deep Sea Points Found",
            )

    # 2. 9种微观策略扫描
    if args.part in ("all", "sampling"):
        sampling_results = run_sampling_sweep(mode)
        if sampling_results:
            save_csv(sampling_results, "stage1_sampling_summary.csv")
            plot_comparison(
                sampling_results,
                "stage1_sampling_comparison.png",
                "Stage1: Micro-sampling Strategy Comparison (9 Methods)",
                "Average Deep Sea Points Found",
            )

    # 3. 合并汇总（需求第十四节：stage1_summary.csv）
    all_results = temp_results + sampling_results
    if all_results:
        save_csv(all_results, "stage1_summary.csv")
        print("\n" + "=" * 96)
        print("  Stage1 消融实验全部完成")
        print("=" * 96)
        print(f"  {'配置':<24} {'平均深海点':>12} {'标准差':>10} {'平均查询':>10} "
              f"{'成功率(完成度)':>16} {'跑满目标次数':>14}")
        print("-" * 96)
        for r in all_results:
            target = r.get("target_deep_sea_points") or 40
            rate = r.get("success_rate")
            if rate is None:
                rate = (r["mean_points"] / target) if target else 0.0
            reached = r.get("target_reached_count", 0)
            n_runs = r.get("n_runs", 0)
            print(f"  {r['label']:<24} {r['mean_points']:>12.2f} {r['std_points']:>10.2f} "
                  f"{r['mean_queries']:>10.1f} "
                  f"{r['mean_points']:>7.2f}/{target} ({rate * 100:>5.1f}%) "
                  f"{reached:>8}/{n_runs}")
        print("=" * 96)
        print("  成功率 = 平均深海点数 / 目标深海点数（完成度口径，便于横向对比各配置）")
        print("\n  下一步：依据本表确定 Stage1 最优感知方式，将其填入 Stage2 配置的非测试变量")
    else:
        print("\n[失败] 未找到任何有效配置，请检查 configs/ablation/stage1_perception/")
        sys.exit(1)


if __name__ == "__main__":
    main()
