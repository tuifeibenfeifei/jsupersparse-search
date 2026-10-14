"""
run_ablation_stage4.py
Stage4 超参数微调层消融实验调度脚本

【职责】对应 需求.txt 第十四节 run_ablation_stage4.py：
    读取 stage4_hyperparams/ 下的 YAML（继承前三阶段最优）→ 循环调用
    main_algorithm 跑 20种子×20次 → 生成 CSV 和【收敛曲线】→ 存入 results/ablation_logs/

【消融变量】需求第八节 Stage4（超参数微调层）：
    锁定前三阶段最优结构，仅测试【好奇心权重数值】与【查询次数/目标数量】。

【需求第十节专项】query_target_sweep 运行时【必须】输出标准的
    「查询次数 - 累计发现点数」CSV 数据，用于绘制收敛曲线图。
    本脚本从 run_experiment 返回的 raw_results[].convergence_data 提取每次运行的
    收敛轨迹，跨 400 次运行按查询次数分箱聚合平均，得到平滑收敛曲线。
"""

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ablation_common as C

STAGE4_CONFIG_DIR = C.PROJECT_ROOT / "configs" / "ablation" / "stage4_hyperparams"


def run_curiosity_weight_sweep(mode: str) -> List[Dict[str, float]]:
    """好奇心权重扫描 [1,3,5]，label 形如 curiosity_weight=3"""
    path = STAGE4_CONFIG_DIR / "curiosity_weight_sweep.yaml"
    if not path.exists():
        print(f"[跳过] 配置文件不存在: {path}")
        return []
    return C.run_sweep(
        C.load_yaml(path), sweep_field="curiosity_weight_list", mode=mode,
        stage_title="Stage4", label_prefix="curiosity_weight=",
        target_key="curiosity_weight",
    )


def _aggregate_convergence(raw_results: List[dict], n_bins: int = 100) -> List[Dict]:
    """
    把多次运行的收敛轨迹聚合成一条平滑曲线。

    每次运行的 convergence_data 是 [(query_count, deep_sea_points, stage), ...]。
    不同运行的查询步点不完全对齐，故按查询次数等宽分箱（n_bins 段），
    对落在每箱内的 (query_count, deep_sea_points) 取均值，得到跨运行的平均收敛轨迹。

    Returns: [{"query_count": 箱中心, "deep_sea_points": 平均累计点数}, ...]
    """
    if not raw_results:
        return []

    # 收集所有 (query_count, deep_sea_points) 采样点
    samples: List[tuple] = []
    max_q = 0
    for r in raw_results:
        for d in r.get("convergence_data", []):
            q = d.get("query_count", 0)
            p = d.get("deep_sea_points", 0)
            samples.append((q, p))
            if q > max_q:
                max_q = q
    if not samples or max_q <= 0:
        return []

    bin_width = max_q / n_bins
    sum_p = [0.0] * n_bins
    sum_q = [0.0] * n_bins
    count = [0] * n_bins
    for q, p in samples:
        idx = min(int(q / bin_width), n_bins - 1)
        sum_p[idx] += p
        sum_q[idx] += q
        count[idx] += 1

    curve = []
    for i in range(n_bins):
        if count[i] > 0:
            curve.append({
                "query_count": sum_q[i] / count[i],
                "deep_sea_points": sum_p[i] / count[i],
            })
    return curve


def run_query_target_sweep(mode: str):
    """
    查询预算/目标配对扫描 + 收敛曲线导出（需求第十节专项）。
    query_budget_list 与 target_list 按索引配对，逐对运行。
    Returns: (汇总统计列表, 收敛曲线字典 {label: 聚合轨迹})
    """
    path = STAGE4_CONFIG_DIR / "query_target_sweep.yaml"
    if not path.exists():
        print(f"[跳过] 配置文件不存在: {path}")
        return [], {}

    yaml_config = C.load_yaml(path)
    budgets = yaml_config.get("query_budget_list", [])
    targets = yaml_config.get("target_list", [])
    if not budgets:
        print("[跳过] query_budget_list 为空")
        return [], {}
    # 目标列表缺省或长度不符时，统一以 40 为默认目标（需求第十三节）
    if len(targets) != len(budgets):
        targets = [40] * len(budgets)

    shared = {k: v for k, v in yaml_config.items()
              if k not in ("query_budget_list", "target_list")}

    results: List[Dict[str, float]] = []
    curves: Dict[str, List[Dict]] = {}
    for budget, target in zip(budgets, targets):
        overrides = {"max_queries_per_episode": budget, "target_deep_sea_points": target}
        config = C.build_config(shared, overrides, mode)
        label = f"budget={budget}"
        stats = C.run_single_config(config, label=label, stage_title="Stage4")
        results.append(stats)
        # 从原始结果提取并聚合收敛曲线（需求第十节）
        curves[label] = _aggregate_convergence(stats.get("raw_results", []))

    return results, curves


def main():
    args = C.make_arg_parser(
        "Stage4 超参数微调层消融实验调度脚本",
        part_choices=["all", "curiosity", "budget"],
    ).parse_args()
    mode = args.mode

    C.print_stage_banner(
        "Stage4 超参数微调层消融实验", mode,
        "好奇心权重 [1,3,5] + 查询预算扫描(收敛曲线)，继承前三阶段最优",
    )

    curiosity_results: List[Dict[str, float]] = []
    budget_results: List[Dict[str, float]] = []
    curves: Dict[str, List[Dict]] = {}

    # 1. 好奇心权重扫描
    if args.part in ("all", "curiosity"):
        curiosity_results = run_curiosity_weight_sweep(mode)
        if curiosity_results:
            C.save_csv(curiosity_results, "stage4_curiosity_summary.csv")
            C.plot_comparison(curiosity_results, "stage4_curiosity_comparison.png",
                              "Stage4: Curiosity Weight Sweep [1,3,5]")

    # 2. 查询预算扫描 + 收敛曲线（需求第十节）
    if args.part in ("all", "budget"):
        budget_results, curves = run_query_target_sweep(mode)
        if budget_results:
            C.save_csv(budget_results, "stage4_budget_summary.csv")
            C.plot_comparison(budget_results, "stage4_budget_comparison.png",
                              "Stage4: Query Budget Sweep")
        if curves:
            # 需求第十节：标准的「查询次数-累计发现点数」CSV + 收敛曲线图
            C.save_convergence_csv(curves, "stage4_convergence.csv")
            C.plot_convergence_curves(curves, "stage4_convergence_curves.png",
                                      "Stage4: Convergence Curves (Query Count vs Deep Sea Points)")

    # 3. 合并汇总（需求第十四节：stage4_summary.csv）
    all_results = curiosity_results + budget_results
    if all_results:
        C.save_csv(all_results, "stage4_summary.csv")
        C.print_summary(all_results, "Stage4",
                        next_hint="下一步：依据本表与收敛曲线确定最终超参数，"
                                  "将 Stage1~4 全部最优组合填入 configs/final_model.yaml")
    else:
        print("\n[失败] 未找到任何有效配置，请检查 configs/ablation/stage4_hyperparams/")
        sys.exit(1)


if __name__ == "__main__":
    main()
