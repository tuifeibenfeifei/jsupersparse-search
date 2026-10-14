"""
icm.py
[ICM 基线] Intrinsic Curiosity Module（Pathak et al., 2017 的简化适配版）。

核心机制：
    前向模型 f(s, a) -> ŝ'，内在奖励 = 前向预测误差 ||ŝ' - s'||²。
    误差越大说明该转移越"出乎意料"，鼓励 agent 探索无法预测的区域，
    从而克服稀疏外部奖励下的探索不足问题。

与主框架的差异：
    · 无温度导向偏向（温度仅作为前向模型输入特征之一）
    · 内在奖励来自预测误差，外在奖励为深海发现(+10) / 越界(-10) / 重复(-5)
    · 只依赖 L0 + 本文件自带的 numpy 网络，不依赖 main_algorithm.py

参考文献：Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017).
    Curiosity-driven Exploration by Self-supervised Prediction. ICML.
"""

import numpy as np
from typing import Dict, List, Optional

from ._common import MLP, make_action_onehot
from ._intrinsic_agent import IntrinsicModule, run_intrinsic_rl


class _ICMModule(IntrinsicModule):
    """ICM 内在奖励模块：前向模型预测误差"""

    def __init__(self, config: dict, seed: Optional[int] = None):
        # 状态特征维度 3 + 动作 one-hot 维度 num_actions
        num_directions = int(config.get("action_directions", 8))
        action_steps = list(config.get("action_steps", [1, 2, 4, 8]))
        num_actions = num_directions * len(action_steps)
        self._num_actions = num_actions
        hidden = int(config.get("icm_hidden_size", 32))
        lr = float(config.get("icm_learning_rate", 0.01))
        self._forward = MLP(in_dim=3 + num_actions, hidden=hidden,
                            out_dim=3, lr=lr, seed=seed)
        self._last_error = 0.0

    def intrinsic(self, cur_feat, action_idx: int, nxt_feat) -> float:
        x = np.concatenate([np.asarray(cur_feat),
                            make_action_onehot(action_idx, self._num_actions)])
        pred = self._forward.predict(x)
        self._last_error = float(np.mean((pred - np.asarray(nxt_feat)) ** 2))
        return self._last_error

    def train(self, cur_feat, action_idx: int, nxt_feat) -> None:
        x = np.concatenate([np.asarray(cur_feat),
                            make_action_onehot(action_idx, self._num_actions)])
        self._forward.train_step(x, np.asarray(nxt_feat))


def run_icm(config: dict, seed: int) -> dict:
    """
    执行一次 ICM 基线 episode。

    Args:
        config: 配置字典（含 RL 超参、icm_hidden_size / icm_learning_rate /
                intrinsic_weight 等 ICM 专属参数）
        seed: 世界种子；算法随机性由 config["algo_seed"]（缺省取 seed）驱动
    Returns:
        结果字典（格式与 run_single_episode 一致）
    """
    module = _ICMModule(config, seed=int(config.get("algo_seed", seed)))
    return run_intrinsic_rl(config, seed, module)
