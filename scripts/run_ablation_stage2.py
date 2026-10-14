"""
run_ablation_stage2.py
Stage2 宏观移动层消融实验调度脚本

【职责】对应 需求.txt 第十四节 run_ablation_stage2.py：
    读取 stage2_movement/ 下的 YAML（继承 Stage1 最优）→ 循环调用 main_algorithm
    跑 20种子×20次 → 生成 CSV 和图表 → 存入 results/ablation_logs/

【消融变量】需求第八节 Stage2（宏观移动层）：
    锁定 Stage1 最优结果，仅测试【宏观遍历策略】与【深海点间距阈值】。

【两组实验】
    1. finder_sweep   : 5 种宏观遍历策略对比（ray/spiral/adaptive/grid/anchor）
    2. dist_threshold : 间距阈值 [416,448,480,512] 扫描（锁定单一 finder，避免变量耦合）
"""

import sys
from pathlib import Path
from typing import Dict, List

# 复用公共调度模块（YAML加载/run_experiment驱动/CSV/绘图/汇总）
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ablation_common as C

STAGE2_CONFIG_DIR = C.PROJECT_ROOT / "configs" / "ablation" / "stage2_movement"


def run_finder_sweep(mode: str) -> List[Dict[str, float]]:
    """宏观遍历策略扫描（5 种），label 形如 finder=grid_scan"""
    path = STAGE2_CONFIG_DIR / "finder_sweep.yaml"
    if not path.exists():
        print(f"[跳过] 配置文件不存在: {path}")
        return []
    return C.run_sweep(
        C.load_yaml(path), sweep_field="chunk_finder_list", mode=mode,
        stage_title="Stage2", label_prefix="finder=", target_key="chunk_finder",
    )


def run_distance_threshold_sweep(mode: str) -> List[Dict[str, float]]:
    """间距阈值扫描 [416,448,480,512]，label 形如 threshold=416"""
    path = STAGE2_CONFIG_DIR / "dist_threshold_sweep.yaml"
    if not path.exists():
        print(f"[跳过] 配置文件不存在: {path}")
        return []
    return C.run_sweep(
        C.load_yaml(path), sweep_field="distance_threshold_list", mode=mode,
        stage_title="Stage2", label_prefix="threshold=",
        target_key="deep_sea_distance_threshold",
    )


def main():
    args = C.make_arg_parser(
        "Stage2 宏观移动层消融实验调度脚本",
        part_choices=["all", "finder", "threshold"],
    ).parse_args()
    mode = args.mode

    C.print_stage_banner(
        "Stage2 宏观移动层消融实验", mode,
        "宏观遍历策略 5 种 + 深海点间距阈值 [416,448,480,512]（继承 Stage1 最优）",
    )

    finder_results: List[Dict[str, float]] = []
    threshold_results: List[Dict[str, float]] = []

    # 1. 宏观遍历策略扫描
    if args.part in ("all", "finder"):
        finder_results = run_finder_sweep(mode)
        if finder_results:
            C.save_csv(finder_results, "stage2_finder_summary.csv")
            C.plot_comparison(finder_results, "stage2_finder_comparison.png",
                              "Stage2: Macro-traversal Strategy Comparison (5 Methods)")

    # 2. 间距阈值扫描
    if args.part in ("all", "threshold"):
        threshold_results = run_distance_threshold_sweep(mode)
        if threshold_results:
            C.save_csv(threshold_results, "stage2_threshold_summary.csv")
            C.plot_comparison(threshold_results, "stage2_threshold_comparison.png",
                              "Stage2: Deep-sea Distance Threshold Sweep [416,448,480,512]")

    # 3. 合并汇总（需求第十四节：stage2_summary.csv）
    all_results = finder_results + threshold_results
    if all_results:
        C.save_csv(all_results, "stage2_summary.csv")
        C.print_summary(all_results, "Stage2",
                        next_hint="下一步：依据本表确定 Stage2 最优移动骨架（遍历策略+间距阈值），"
                                  "将其填入 Stage3 配置的非测试变量")
    else:
        print("\n[失败] 未找到任何有效配置，请检查 configs/ablation/stage2_movement/")
        sys.exit(1)


if __name__ == "__main__":
    main()
