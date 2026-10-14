"""
run_ablation_stage3.py
Stage3 奖励塑形层消融实验调度脚本

【职责】对应 需求.txt 第十四节 run_ablation_stage3.py：
    读取 stage3_reward/ 下的 4 个 YAML（继承前两阶段最优）→ 循环调用
    main_algorithm 跑 20种子×20次 → 生成 CSV 和图表 → 存入 results/ablation_logs/

【消融变量】需求第八节 Stage3（奖励塑形层）：
    锁定前两阶段最优结果，运行 4 个奖励模块文件的消融
    （关深海 / 关好奇 / 关惩罚 / 全开），确定哪些奖励信号是必要的。
"""

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ablation_common as C

STAGE3_CONFIG_DIR = C.PROJECT_ROOT / "configs" / "ablation" / "stage3_reward"

# 4 个奖励消融配置（需求第五节：4 个独立文件，每种各关一次 + 全开对照）
# 顺序即汇总表与柱状图的展示顺序：全开基准在前，便于对比各关闭项的影响。
REWARD_CONFIGS = [
    ("reward_all_on", "all_on(全开对照)"),
    ("reward_no_deep", "no_deep(关深海)"),
    ("reward_no_curiosity", "no_curiosity(关好奇)"),
    ("reward_no_penalty", "no_penalty(关惩罚)"),
]


def run_reward_ablation(mode: str) -> List[Dict[str, float]]:
    """运行 4 种奖励开关组合，每个对应 stage3_reward/ 下一个独立 YAML"""
    results = []
    for cfg_name, label in REWARD_CONFIGS:
        path = STAGE3_CONFIG_DIR / f"{cfg_name}.yaml"
        if not path.exists():
            print(f"[跳过] 配置文件不存在: {path}")
            continue
        yaml_config = C.load_yaml(path)
        # 本阶段无 sweep 列表，每个 YAML 即一个完整单配置
        config = C.build_config(yaml_config, {}, mode)
        results.append(C.run_single_config(config, label=label, stage_title="Stage3"))
    return results


def main():
    args = C.make_arg_parser("Stage3 奖励塑形层消融实验调度脚本").parse_args()
    mode = args.mode

    C.print_stage_banner(
        "Stage3 奖励塑形层消融实验", mode,
        "4 个奖励模块开关（关深海/关好奇/关惩罚/全开），继承前两阶段最优",
    )

    results = run_reward_ablation(mode)
    if not results:
        print("\n[失败] 未找到任何有效配置，请检查 configs/ablation/stage3_reward/")
        sys.exit(1)

    C.save_csv(results, "stage3_summary.csv")
    C.plot_comparison(
        results, "stage3_comparison.png",
        "Stage3: Reward Module Ablation (4 Configurations)",
    )
    C.print_summary(
        results, "Stage3",
        next_hint="下一步：对比全开基准与各关闭项，确定哪些奖励信号是必要的，"
                  "将必要奖励组合 + 前两阶段最优填入 Stage4 配置",
    )


if __name__ == "__main__":
    main()
