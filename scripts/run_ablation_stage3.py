"""
run_ablation_stage3.py
Stage3 reward shaping layer ablation experiment scheduling script

[Responsibility] Corresponds to Req. Sec. 14 in requirements.txt run_ablation_stage3.py:
    Read 4 YAML files under stage3_reward/ (inheriting the optimum of the first two stages) → loop calling
    main_algorithm for 20 seeds × 20 repetitions → generate CSV and plots → store in results/ablation_logs/

[Ablation variables] Req. Sec. 8 Stage3 (reward shaping layer):
    Lock the optimal results of the first two stages; run ablation of 4 reward module files
    (deep-sea off / curiosity off / penalty off / all on) to determine which reward signals are necessary.
"""

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ablation_common as C

STAGE3_CONFIG_DIR = C.PROJECT_ROOT / "configs" / "ablation" / "stage3_reward"

# 4 reward ablation configs (Req. Sec. 5: 4 independent files, each turned off once + all-on control)
# Order is the display order in the summary table and bar chart: all-on baseline first for easy comparison of each off-item's effect.
REWARD_CONFIGS = [
    ("reward_all_on", "all_on (all-on control)"),
    ("reward_no_deep", "no_deep (deep-sea off)"),
    ("reward_no_curiosity", "no_curiosity (curiosity off)"),
    ("reward_no_penalty", "no_penalty (penalty off)"),
]


def run_reward_ablation(mode: str) -> List[Dict[str, float]]:
    """Run 4 reward switch combinations, each corresponding to an independent YAML under stage3_reward/"""
    results = []
    for cfg_name, label in REWARD_CONFIGS:
        path = STAGE3_CONFIG_DIR / f"{cfg_name}.yaml"
        if not path.exists():
            print(f"[Skip] Config file not found: {path}")
            continue
        yaml_config = C.load_yaml(path)
        # This stage has no sweep list; each YAML is a complete single configuration
        config = C.build_config(yaml_config, {}, mode)
        results.append(C.run_single_config(config, label=label, stage_title="Stage3"))
    return results


def main():
    C.require_real_cubiomes()   # refuse to run on Mock (missing cubiomes.dll)
    args = C.make_arg_parser("Stage3 reward shaping layer ablation experiment scheduling script").parse_args()
    mode = args.mode

    C.print_stage_banner(
        "Stage3 reward shaping layer ablation experiment", mode,
        "4 reward module switches (deep-sea off / curiosity off / penalty off / all on), inheriting the optimum of the first two stages",
    )

    results = run_reward_ablation(mode)
    if not results:
        print("\n[Failed] No valid configuration found. Please check configs/ablation/stage3_reward/")
        sys.exit(1)

    C.save_csv(results, "stage3_summary.csv")
    C.plot_comparison(
        results, "stage3_comparison.png",
        "Stage3: Reward Module Ablation (4 Configurations)",
    )
    C.print_summary(
        results, "Stage3",
        next_hint="Next step: Compare the all-on baseline with each off-item to determine which reward signals are necessary, "
                  "then fill the necessary reward combination + the optimum of the first two stages into the Stage4 config",
    )


if __name__ == "__main__":
    main()
