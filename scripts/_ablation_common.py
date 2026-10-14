"""
_ablation_common.py
各 Stage 消融调度脚本的公共基础设施（需求第十五节：调度与持久化层统一口径）。

抽取以下共享逻辑，避免 4 个 stage 脚本各重复 ~150 行：
    · YAML 加载与单配置构造
    · 调用 DeepSeaExplorer.run_experiment 跑 20种子×20次
    · 统计聚合（mean/std/success_rate）
    · CSV 写入（统一列，含完成度口径成功率）
    · 对比柱状图（带误差棒，需求第十二节）
    · 收敛曲线绘图（Stage4 专用，需求第十节）
    · 汇总表打印

⚠️ 本模块不含任何消融变量数值，全部由调用方传入的 YAML 驱动（需求第十节）。
"""

import sys
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Callable

import yaml
import matplotlib
matplotlib.use("Agg")   # 无窗口环境下也能出图
import matplotlib.pyplot as plt

# 将项目根目录加入路径，使 `import src.*` 可用
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main_algorithm import DeepSeaExplorer

# ==================== 实验常量（需求第七节 / 第十二节）====================
WORLD_SEEDS = list(range(1, 21))    # 所有实验在种子 1~20 上运行
REPEATS = 20                        # 同一种子重复 20 次取平均
TOTAL_RUNS = len(WORLD_SEEDS) * REPEATS  # 单配置总运行次数（400）

RESULTS_DIR = PROJECT_ROOT / "results" / "ablation_logs"

# 配置总目录（供 run_finder_showdown.py 等脚本定位 YAML 配置；需求第十节：配置为唯一数值来源）
CONFIG_DIR = PROJECT_ROOT / "configs"

# CSV 统一列（成功率口径 = 平均深海点数 / 目标深海点数，即完成度）
CSV_FIELDS = ["label", "mean_points", "std_points", "mean_queries",
              "std_queries", "success_rate", "target_reached_count", "n_runs"]


def load_yaml(path: Path) -> dict:
    """安全加载 YAML 配置（空文件返回空字典）"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_config(base_config: dict, overrides: dict, mode: str) -> dict:
    """
    构造单配置：以 YAML 为唯一数值来源（需求第十节），仅注入输出模式与本配置的
    扫描变量覆盖项。不硬编码任何消融数值。
    """
    config = dict(base_config)
    config["output_mode"] = mode
    for key, value in overrides.items():
        config[key] = value
    return config


def run_single_config(config: dict, label: str, stage_title: str = "Stage") -> Dict[str, float]:
    """
    对单个配置执行 20 世界种子 × 20 次重复实验（需求第十二节：分离环境/算法随机性）。
    Returns: 含 mean/std/success_rate 与原始结果的统计字典（追加 label）。
    """
    print(f"\n{'=' * 74}")
    print(f"▶ {stage_title} 消融配置: {label}")
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


def run_sweep(yaml_config: dict, sweep_field: str, mode: str,
              stage_title: str, label_prefix: str = "",
              target_key: Optional[str] = None) -> List[Dict[str, float]]:
    """
    通用扫描：读取 YAML 中的列表字段 sweep_field，对列表每个值生成一个单配置并运行。
    列表字段本身从共享配置中剔除（需求第九节：sweep 文件不可直接传入 Explorer）。

    Args:
        yaml_config:  已加载的 sweep YAML
        sweep_field:  存放待扫描值的列表键名（如 "micro_strategy_list"）
        mode:         simple / detailed
        stage_title:  打印用阶段名
        label_prefix: 结果 label 前缀（如 "finder=" / "threshold="）
        target_key:   扫描值注入的配置键名。缺省时按约定取 sweep_field 去掉 "_list"
                      后缀（micro_strategy_list -> micro_strategy）。当配置键与列表键
                      不同名时（distance_threshold_list -> deep_sea_distance_threshold）
                      必须显式传入。
    Returns:
        每个扫描值一条统计结果的列表
    """
    values = yaml_config.get(sweep_field, [])
    if not values:
        print(f"[跳过] {sweep_field} 为空或不存在")
        return []

    if target_key is None:
        target_key = sweep_field[:-5] if sweep_field.endswith("_list") else sweep_field

    shared = {k: v for k, v in yaml_config.items() if k != sweep_field}
    results = []
    for v in values:
        config = build_config(shared, {target_key: v}, mode)
        label = f"{label_prefix}{v}"
        results.append(run_single_config(config, label, stage_title))
    return results


def save_csv(results: List[Dict[str, float]], filename: str,
             out_dir: Optional[Path] = None) -> Optional[Path]:
    """保存统计结果到 CSV（需求第十四节：存入 results/ 对应目录）"""
    if not results:
        return None
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in CSV_FIELDS})
    print(f"[保存] CSV 已写入: {filepath}")
    return filepath


def plot_comparison(results: List[Dict[str, float]], filename: str,
                    title: str, ylabel: str = "Average Deep Sea Points Found",
                    out_dir: Optional[Path] = None) -> Optional[Path]:
    """生成带误差棒的对比柱状图（需求第十二节：误差棒证明结果非偶然）"""
    if not results:
        return None
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename

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


def plot_convergence_curves(curves: Dict[str, List[Dict]], filename: str,
                            title: str = "Stage4: Convergence Curves",
                            out_dir: Optional[Path] = None) -> Optional[Path]:
    """
    绘制「查询次数 - 累计发现点数」收敛曲线（需求第十节：Stage4 收敛曲线数据）。

    Args:
        curves: {配置label: convergence_data 列表}，每条为
                {"query_count":int, "deep_sea_points":int, "stage":int}
    """
    if not curves:
        return None
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, data in curves.items():
        if not data:
            continue
        xs = [d["query_count"] for d in data]
        ys = [d["deep_sea_points"] for d in data]
        ax.plot(xs, ys, label=label, linewidth=1.5)
    ax.set_xlabel("Query Count")
    ax.set_ylabel("Cumulative Deep Sea Points")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"[保存] 收敛曲线已写入: {filepath}")
    return filepath


def save_convergence_csv(curves: Dict[str, List[Dict]], filename: str,
                         out_dir: Optional[Path] = None) -> Optional[Path]:
    """
    导出收敛曲线原始数据为 CSV（需求第十节：标准的"查询次数-累计发现点数"CSV）。
    长表格式：label, query_count, deep_sea_points, stage
    """
    if not curves:
        return None
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "query_count", "deep_sea_points", "stage"])
        for label, data in curves.items():
            for d in data:
                writer.writerow([label, d.get("query_count"),
                                 d.get("deep_sea_points"), d.get("stage")])
    print(f"[保存] 收敛CSV已写入: {filepath}")
    return filepath


def print_summary(all_results: List[Dict[str, float]], stage_title: str,
                  next_hint: str = "") -> None:
    """打印消融汇总大表（成功率 = 平均深海点 / 目标，完成度口径）"""
    if not all_results:
        print(f"\n[失败] 未找到任何有效配置")
        return
    print("\n" + "=" * 96)
    print(f"  {stage_title} 消融实验全部完成")
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
    if next_hint:
        print(f"\n  {next_hint}")


def make_arg_parser(description: str, part_choices: Optional[List[str]] = None):
    """统一命令行参数：--mode 简易/详细，--part 运行范围"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--mode", choices=["simple", "detailed"], default="simple",
                        help="输出模式：simple=简易版(20行平均值)，detailed=详细版(20次完整结果+实时查询输出)")
    if part_choices:
        parser.add_argument("--part", choices=part_choices, default="all",
                            help="运行范围")
    return parser


def print_stage_banner(stage_title: str, mode: str, ablation_vars: str):
    """打印各 stage 脚本的统一开场横幅"""
    print("=" * 74)
    print(f"  {stage_title}")
    print("=" * 74)
    print(f"  实验规模    : 世界种子 {WORLD_SEEDS[0]}~{WORLD_SEEDS[-1]} "
          f"× 每种子重复 {REPEATS} 次 = 单配置 {TOTAL_RUNS} 次运行")
    print(f"  输出模式    : {'详细版 (20次完整结果 + 实时查询输出)' if mode == 'detailed' else '简易版 (20行平均值)'}")
    print(f"  消融变量    : {ablation_vars}")
    print(f"  输出目录    : {RESULTS_DIR}")
    print("=" * 74)
