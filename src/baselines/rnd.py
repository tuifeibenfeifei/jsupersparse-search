"""
rnd.py
[RND 基线] Random Network Distillation（Burda et al., 2018 的简化适配版）。

核心机制：
    固定随机初始化的目标网络 T(s)（不再训练），预测器网络 P(s) 通过监督学习
    拟合 T(s)。内在奖励 = ||T(s') - P(s')||²。
    由于 T 固定，训练过的状态 P 能拟合、误差趋近 0；未训练的新状态 P 拟合差、
    误差大，从而天然形成"新颖性"信号，驱动 agent 探索陌生区域。

与主框架的差异：
    · 无温度导向偏向（温度仅作为状态特征之一）
    · 内在奖励来自 RND 蒸馏误差，外在奖励为深海发现(+10) / 越界(-10) / 重复(-5)
    · 只依赖 L0 + 本文件自带的 numpy 网络，不依赖 main_algorithm.py

参考文献：Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2018).
    Exploration by Random Network Distillation. ICLR.
"""

import numpy as np
from typing import Dict, List, Optional

from ._common import MLP
from ._intrinsic_agent import IntrinsicModule, run_intrinsic_rl


class _RNDModule(IntrinsicModule):
    """RND 内在奖励模块：预测器 vs 固定目标网络 的蒸馏误差"""

    def __init__(self, config: dict, seed: Optional[int] = None):
        hidden = int(config.get("rnd_hidden_size", 32))
        lr = float(config.get("rnd_learning_rate", 0.01))
        # 状态特征维度 3（区块坐标归一化 + 温度）
        rng_seed = seed if seed is not None else 0
        self._target = MLP(in_dim=3, hidden=hidden, out_dim=3,
                           lr=lr, seed=rng_seed + 1)   # 固定目标网络，不训练
        self._predictor = MLP(in_dim=3, hidden=hidden, out_dim=3,
                              lr=lr, seed=rng_seed + 2)
        self._last_error = 0.0

    def intrinsic(self, cur_feat, action_idx: int, nxt_feat) -> float:
        t = self._target.predict(nxt_feat)
        p = self._predictor.predict(nxt_feat)
        self._last_error = float(np.mean((t - p) ** 2))
        return self._last_error

    def train(self, cur_feat, action_idx: int, nxt_feat) -> None:
        # 预测器以固定目标网络输出为监督信号（目标网络参数永不更新）
        t = self._target.predict(nxt_feat)
        self._predictor.train_step(nxt_feat, t)


def run_rnd(config: dict, seed: int) -> dict:
    """
    执行一次 RND 基线 episode。

    Args:
        config: 配置字典（含 RL 超参、rnd_hidden_size / rnd_learning_rate /
                intrinsic_weight 等 RND 专属参数）
        seed: 世界种子；算法随机性由 config["algo_seed"]（缺省取 seed）驱动
    Returns:
        结果字典（格式与 run_single_episode 一致）
    """
    module = _RNDModule(config, seed=int(config.get("algo_seed", seed)))
    return run_intrinsic_rl(config, seed, module)
