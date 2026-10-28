"""
_baselines_common.py
Baseline algorithm common utilities (depends only on L0 infrastructure layer: cubiomes_wrapper / database)

Req. Sec. 11 L4 key constraints:
    - Baseline algorithms are independent of main_algorithm.py, do not depend on L2/L1, only on L0
    - Each algorithm has its own independent execution loop
    - Return format is consistent with main_algorithm.run_single_episode() result dict,
      so scripts/run_baselines.py can uniformly aggregate and plot

This module provides:
    - make_env()            construct environment (CubiomesWrapper + DeepSeaDatabase)
    - query_chunk_center()  perform one standard query on a single chunk (temperature probe + deep-sea determination,
                            2 black-box queries per chunk, consistent with the main framework's center_point micro-strategy)
    - build_result()        construct a result dict isomorphic to run_single_episode
    - MLP                   lightweight two-layer network in numpy (for ICM / RND intrinsic reward)
"""

import numpy as np
from typing import List, Tuple, Optional

from ..cubiomes_wrapper import CubiomesWrapper
from ..database import DeepSeaDatabase

# 8-direction displacement table, consistent with main framework _DIRECTION_DELTAS
DIRECTION_DELTAS: List[Tuple[int, int]] = [
    (1, 0), (1, -1), (0, -1), (-1, -1),
    (-1, 0), (-1, 1), (0, 1), (1, 1),
]


def make_env(config: dict, seed: int) -> Tuple[CubiomesWrapper, DeepSeaDatabase]:
    """Construct black-box environment and database (baselines use only L0)"""
    wrapper = CubiomesWrapper(seed=seed)
    db = DeepSeaDatabase(config.get("deep_sea_distance_threshold", 416))
    return wrapper, db


def chunk_center(cx: int, cz: int) -> Tuple[int, int]:
    """World coordinates of chunk center (consistent with main framework center_point micro-strategy sampling)"""
    return cx * 16 + 8, cz * 16 + 8


def within_radius(cx: int, cz: int, radius: int) -> bool:
    """Whether a chunk is within search range (square region of radius 10000 cells)"""
    return abs(cx * 16) <= radius and abs(cz * 16) <= radius


def query_chunk_center(wrapper: CubiomesWrapper, db: DeepSeaDatabase,
                       config: dict, cx: int, cz: int,
                       temperature_history: Optional[List] = None,
                       temperature_samples: Optional[List] = None,
                       temperature_enabled: bool = True):
    """
    Perform a standard query on chunk (cx, cz):
        1. Query chunk center temperature (1 query, counted in budget)
        2. Determine whether chunk center is deep-sea (1 query, counted in budget)
        3. If deep-sea hit, write to database (with spacing threshold dedup), return whether it is a new valid point

    Returns:
        (is_deep, temp, is_new_valid_point)
    """
    wx, wz = chunk_center(cx, cz)
    try:
        temp = wrapper.get_temperature(wx, wz)
    except Exception:
        temp = None
    try:
        is_deep = wrapper.is_deep_ocean(wx, wz)
    except Exception:
        is_deep = False

    if temperature_history is not None:
        temperature_history.append(temp)
    if temperature_samples is not None and temperature_enabled:
        temperature_samples.append((temp, is_deep))

    if is_deep:
        stored = temp if temp is not None else 0.0
        if db.add_point(wx, wz, stored):
            return is_deep, temp, True
    return is_deep, temp, False


def build_result(seed: int, wrapper: CubiomesWrapper, db: DeepSeaDatabase,
                 convergence_data: List[dict],
                 temperature_history: List = None,
                 temperature_samples: List = None,
                 visited_chunks: int = 0,
                 temperature_probes: int = 0) -> dict:
    """
    Construct a result dict isomorphic to main_algorithm.run_single_episode(),
    so run_baselines.py can reuse the unified aggregation/plotting logic.
    """
    temperature_history = temperature_history or []
    temperature_samples = temperature_samples or []
    points = db.get_all_points()
    return {
        "seed": seed,
        "total_queries": wrapper.query_count,
        "deep_sea_points_found": len(db),
        "deep_sea_points": points,
        "final_stage": 1,          # baselines are a single algorithm, no three-stage concept, fixed as 1
        "epsilon": None,
        "convergence_data": convergence_data,
        "temperature_history": list(temperature_history),
        "temperature_histogram": {},
        "temperature_deep_probability": {},
        "temperature_preference": {},
        "temperature_samples": list(temperature_samples),
        "stage_transitions": [],
        "visited_states": visited_chunks,
        "visited_chunks": visited_chunks,
        "temperature_probes": temperature_probes,
    }


class MLP:
    """
    Lightweight two-layer fully connected network (numpy implementation, SGD training).

    Used only for ICM (forward model error) and RND (predictor vs fixed target network) intrinsic
    curiosity reward computation; does not depend on torch/tf, ensuring zero extra dependencies.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 lr: float = 0.01, seed: Optional[int] = None):
        rng = np.random.RandomState(seed if seed is not None else 0)
        self.W1 = rng.randn(in_dim, hidden) * np.sqrt(2.0 / max(in_dim, 1))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.randn(hidden, out_dim) * np.sqrt(2.0 / max(hidden, 1))
        self.b2 = np.zeros(out_dim)
        self.lr = float(lr)

    def _forward(self, x: np.ndarray):
        h = np.maximum(0.0, x @ self.W1 + self.b1)   # ReLU hidden layer
        return h @ self.W2 + self.b2, h

    def predict(self, x) -> np.ndarray:
        """Forward output (does not update parameters)"""
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        out, _ = self._forward(x)
        return out.reshape(-1)

    def train_step(self, x, y) -> float:
        """Single-step SGD (MSE loss), returns current batch mean squared error"""
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        y = np.asarray(y, dtype=np.float64).reshape(1, -1)
        out, h = self._forward(x)

        d_out = 2.0 * (out - y)
        d_W2 = h.T @ d_out
        d_b2 = d_out.sum(axis=0)
        d_h = d_out @ self.W2.T
        d_h[h <= 0.0] = 0.0
        d_W1 = x.T @ d_h
        d_b1 = d_h.sum(axis=0)

        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1
        return float(np.mean((out - y) ** 2))


def encode_state_features(cx: int, cz: int, temp: Optional[float],
                          scale: float = 625.0) -> np.ndarray:
    """
    Encode chunk coordinates + temperature as a continuous feature vector for ICM/RND networks.
    scale is set to 625 (the number of chunks for search radius 10000 cells), normalizing coordinates to around [-1, 1].
    """
    t = 0.0 if temp is None else float(np.tanh(temp))
    return np.array([cx / scale, cz / scale, t], dtype=np.float64)


def make_action_onehot(action_idx: int, num_actions: int) -> np.ndarray:
    """Action index -> one-hot vector (for ICM forward model input)"""
    a = np.zeros(num_actions, dtype=np.float64)
    a[action_idx % num_actions] = 1.0
    return a
