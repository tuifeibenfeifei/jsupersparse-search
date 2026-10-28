"""
rnd.py
[RND baseline] Random Network Distillation (simplified adaptation of Burda et al., 2018).

Core mechanism:
    Fixed randomly-initialized target network T(s) (no longer trained); predictor network P(s) via supervised learning
    fits T(s). Intrinsic reward = ||T(s') - P(s')||².
    Since T is fixed, trained states are well-fit by P with error approaching 0; untrained new states are poorly fit by P with
    large error, which naturally forms a "novelty" signal driving the agent to explore unfamiliar regions.

Differences from the main framework:
    · No temperature guidance bias (temperature is only one of the state features)
    · Intrinsic reward comes from RND distillation error; extrinsic reward is deep-sea discovery (+10) / out-of-bounds (-10) / repeat (-5)
    · Only depends on L0 + the numpy network built into this file; does not depend on main_algorithm.py

Reference: Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2018).
    Exploration by Random Network Distillation. ICLR.
"""

import numpy as np
from typing import Dict, List, Optional

from ._common import MLP
from ._intrinsic_agent import IntrinsicModule, run_intrinsic_rl


class _RNDModule(IntrinsicModule):
    """RND intrinsic reward module: distillation error between predictor and fixed target network"""

    def __init__(self, config: dict, seed: Optional[int] = None):
        hidden = int(config.get("rnd_hidden_size", 32))
        lr = float(config.get("rnd_learning_rate", 0.01))
        # State feature dimension 3 (normalized chunk coordinates + temperature)
        rng_seed = seed if seed is not None else 0
        self._target = MLP(in_dim=3, hidden=hidden, out_dim=3,
                           lr=lr, seed=rng_seed + 1)   # Fixed target network, not trained
        self._predictor = MLP(in_dim=3, hidden=hidden, out_dim=3,
                              lr=lr, seed=rng_seed + 2)
        self._last_error = 0.0

    def intrinsic(self, cur_feat, action_idx: int, nxt_feat) -> float:
        t = self._target.predict(nxt_feat)
        p = self._predictor.predict(nxt_feat)
        self._last_error = float(np.mean((t - p) ** 2))
        return self._last_error

    def train(self, cur_feat, action_idx: int, nxt_feat) -> None:
        # Predictor uses fixed target network output as supervision (target network parameters are never updated)
        t = self._target.predict(nxt_feat)
        self._predictor.train_step(nxt_feat, t)


def run_rnd(config: dict, seed: int) -> dict:
    """
    Execute one RND baseline episode.

    Args:
        config: Configuration dict (including RL hyperparameters, rnd_hidden_size / rnd_learning_rate /
                intrinsic_weight and other RND-specific parameters)
        seed: World seed; algorithmic randomness is driven by config["algo_seed"] (defaults to seed)
    Returns:
        Result dict (format consistent with run_single_episode)
    """
    module = _RNDModule(config, seed=int(config.get("algo_seed", seed)))
    return run_intrinsic_rl(config, seed, module)
