"""
_intrinsic_agent.py
ICM / RND shared "intrinsic curiosity + Q-learning" episode skeleton.

Req. Sec. 11 L4:
    - Baseline algorithms are independent of main_algorithm.py, depend only on L0 (cubiomes_wrapper + database)
    - Each has its own independent execution loop
    - Return format is consistent with run_single_episode()

Structure:
    Both algorithms share the same exploration loop (epsilon-greedy action selection -> move -> out-of-bounds/dedup handling
    -> chunk query -> intrinsic+extrinsic reward -> Q-learning update); only the intrinsic reward computation differs:
        - ICM: forward model prediction error (state+action -> next_state)
        - RND: predictor vs fixed random target network output mean squared error
    This module extracts the loop; the two algorithm files each provide an intrinsic object.
"""

import random
from typing import Callable, Dict, List, Optional

from ._common import (make_env, within_radius, query_chunk_center,
                      build_result, encode_state_features, make_action_onehot,
                      DIRECTION_DELTAS)


class IntrinsicModule:
    """Intrinsic reward module interface (implemented separately by ICM / RND)"""

    def intrinsic(self, cur_feat, action_idx: int, nxt_feat) -> float:
        """Return intrinsic reward value (pure function; should not update parameters in this step)"""
        raise NotImplementedError

    def train(self, cur_feat, action_idx: int, nxt_feat) -> None:
        """Update internal model parameters with one transition"""
        raise NotImplementedError


def run_intrinsic_rl(config: dict, seed: int,
                     intrinsic_module: IntrinsicModule) -> dict:
    """
    Run one "intrinsic curiosity + Q-learning" episode (shared by ICM/RND).

    Args:
        config: configuration dict (containing RL hyperparameters, reward values, budget, etc.)
        seed: world seed; algorithmic randomness is driven by config["algo_seed"] (defaults to seed)
        intrinsic_module: intrinsic reward module instance (ICM or RND)
    Returns:
        Result dict (format consistent with run_single_episode)
    """
    wrapper, db = make_env(config, seed)
    radius = int(config.get("search_radius", 10000))
    max_queries = int(config.get("max_queries_per_episode", 10000))
    target = int(config.get("target_deep_sea_points", 40))
    temp_enabled = bool(config.get("temperature_enabled", True))

    # --- RL hyperparameters (config-driven) ---
    lr = float(config.get("learning_rate", 0.1))
    gamma = float(config.get("discount_factor", 0.95))
    eps_start = float(config.get("epsilon_start", 1.0))
    eps_end = float(config.get("epsilon_end", 0.05))
    eps_decay = float(config.get("epsilon_decay", 0.995))
    num_directions = int(config.get("action_directions", 8))
    action_steps = list(config.get("action_steps", [1, 2, 4, 8]))
    num_actions = num_directions * len(action_steps)

    # --- Reward values (config-driven) ---
    deep_reward = float(config.get("deep_sea_reward_value", 10.0))
    oob_penalty = float(config.get("out_of_bounds_penalty", -10.0))
    repeat_penalty = float(config.get("repeat_visit_penalty", -5.0))
    intrinsic_weight = float(config.get("intrinsic_weight", 1.0))

    algo_seed = int(config.get("algo_seed", seed))
    rng = random.Random((seed * 1000003 + algo_seed) & 0xFFFFFFFF)

    # --- State space: coordinate coarsening (chunks // 4), temperature not encoded (consistent with main framework's final architecture) ---
    q_table: Dict[tuple, List[float]] = {}
    visited_states = set()
    visited_chunks = set()

    def get_q(state):
        if state not in q_table:
            q_table[state] = [0.0] * num_actions
        return q_table[state]

    def select_action(state, eps):
        if rng.random() < eps:
            return rng.randint(0, num_actions - 1)
        qv = get_q(state)
        best = max(qv)
        return rng.choice([i for i, q in enumerate(qv) if q == best])

    def state_of(cx, cz):
        return (cx // 4, cz // 4)

    epsilon = eps_start
    current = (0, 0)
    current_state = state_of(0, 0)
    visited_states.add(current_state)
    visited_chunks.add((0, 0))

    convergence: List[dict] = []
    temperature_history: List[float] = []
    temperature_samples: List[tuple] = []
    temperature_probes = 0
    last_experience = None
    step_index = 0
    cur_temp = None

    while wrapper.query_count < max_queries and len(db) < target:
        step_index += 1
        action = select_action(current_state, epsilon)
        dir_idx = action % num_directions
        step_idx = action // num_directions
        step = action_steps[step_idx]
        dx, dz = DIRECTION_DELTAS[dir_idx]

        next_cx = current[0] + dx * step
        next_cz = current[1] + dz * step
        next_state = state_of(next_cx, next_cz)
        nxt_feat = encode_state_features(next_cx, next_cz, cur_temp)

        # --- Out-of-bounds handling ---
        if not within_radius(next_cx, next_cz, radius):
            reward = oob_penalty
            next_state = state_of(0, 0)
            target_chunk = (0, 0)
            done_move = True
        elif (next_cx, next_cz) in visited_chunks:
            # Repeat visit: does not consume query budget, only penalized
            reward = repeat_penalty
            target_chunk = (next_cx, next_cz)
            done_move = True
        else:
            visited_chunks.add((next_cx, next_cz))
            target_chunk = (next_cx, next_cz)
            done_move = False

        # --- Intrinsic reward (based on target state features) ---
        if not done_move:
            cur_feat = encode_state_features(current[0], current[1], cur_temp)
            r_int = intrinsic_module.intrinsic(cur_feat, action, nxt_feat)
            intrinsic_module.train(cur_feat, action, nxt_feat)
            reward = intrinsic_weight * r_int
        # Out-of-bounds/repeat: remain pure penalty, no intrinsic reward

        # --- Chunk query (when not out-of-bounds and not repeat) ---
        found = False
        if not done_move:
            cx, cz = target_chunk
            temperature_probes += 1
            _, temp, found = query_chunk_center(
                wrapper, db, config, cx, cz,
                temperature_history=temperature_history,
                temperature_samples=temperature_samples,
                temperature_enabled=temp_enabled,
            )
            cur_temp = temp
            if found:
                reward += deep_reward

        # --- Q-learning update (non-terminal step, with gamma bootstrapping) ---
        qv = get_q(current_state)
        nq = get_q(next_state)
        td_target = reward + gamma * max(nq)
        qv[action] += lr * (td_target - qv[action])

        current = target_chunk
        current_state = next_state
        visited_states.add(current_state)
        epsilon = max(eps_end, epsilon * eps_decay)

        convergence.append({
            "query_count": wrapper.query_count,
            "deep_sea_points": len(db),
            "stage": 1,
        })

    return build_result(seed, wrapper, db, convergence,
                        temperature_history, temperature_samples,
                        visited_chunks=len(visited_chunks),
                        temperature_probes=temperature_probes)
