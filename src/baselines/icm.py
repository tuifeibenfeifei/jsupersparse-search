"""
icm.py
[ICM baseline] Intrinsic Curiosity Module (simplified adaptation of Pathak et al., 2017).

Core mechanism:
    Forward model f(s, a) -> ŝ', intrinsic reward = forward prediction error ||ŝ' - s'||².
    Larger error means the transition is more "unexpected", encouraging the agent to explore unpredictable regions,
    thereby overcoming the exploration insufficiency problem under sparse extrinsic rewards.

Differences from the main framework:
    · No temperature guidance bias (temperature is only one of the forward model input features)
    · Intrinsic reward comes from prediction error; extrinsic reward is deep-sea discovery (+10) / out-of-bounds (-10) / repeat (-5)
    · Only depends on L0 + the numpy network built into this file; does not depend on main_algorithm.py

Reference: Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017).
    Curiosity-driven Exploration by Self-supervised Prediction. ICML.
"""

import numpy as np
from typing import Dict, List, Optional

from ._common import MLP, make_action_onehot
from ._intrinsic_agent import IntrinsicModule, run_intrinsic_rl


class _ICMModule(IntrinsicModule):
    """ICM intrinsic reward module: forward model prediction error"""

    def __init__(self, config: dict, seed: Optional[int] = None):
        # State feature dimension 3 + action one-hot dimension num_actions
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
    Execute one ICM baseline episode.

    Args:
        config: Configuration dict (including RL hyperparameters, icm_hidden_size / icm_learning_rate /
                intrinsic_weight and other ICM-specific parameters)
        seed: World seed; algorithmic randomness is driven by config["algo_seed"] (defaults to seed)
    Returns:
        Result dict (format consistent with run_single_episode)
    """
    module = _ICMModule(config, seed=int(config.get("algo_seed", seed)))
    return run_intrinsic_rl(config, seed, module)
