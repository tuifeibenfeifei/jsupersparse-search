"""
_intrinsic_agent.py
ICM / RND 共用的「内在好奇心 + Q-learning」episode 骨架。

需求.txt 第十一节 L4：
    · 基线算法独立于 main_algorithm.py，仅依赖 L0（cubiomes_wrapper + database）
    · 各自拥有独立执行循环
    · 返回格式与 run_single_episode() 保持一致

结构说明：
    两个算法共享同一探索循环（epsilon-greedy 动作选择 → 移动 → 越界/去重处理
    → 区块查询 → 内在+外在奖励 → Q-learning 更新），仅内在奖励计算方式不同：
        · ICM：前向模型预测误差（state+action -> next_state）
        · RND：预测器 vs 固定随机目标网络 的输出均方误差
    本模块把循环抽出来，两个算法文件各自提供 intrinsic 对象即可。
"""

import random
from typing import Callable, Dict, List, Optional

from ._common import (make_env, within_radius, query_chunk_center,
                      build_result, encode_state_features, make_action_onehot,
                      DIRECTION_DELTAS)


class IntrinsicModule:
    """内在奖励模块接口（ICM / RND 各自实现）"""

    def intrinsic(self, cur_feat, action_idx: int, nxt_feat) -> float:
        """返回内在奖励值（纯函数，不应在此步更新参数）"""
        raise NotImplementedError

    def train(self, cur_feat, action_idx: int, nxt_feat) -> None:
        """用一次转移更新内部模型参数"""
        raise NotImplementedError


def run_intrinsic_rl(config: dict, seed: int,
                     intrinsic_module: IntrinsicModule) -> dict:
    """
    执行一次「内在好奇心 + Q-learning」episode（ICM/RND 共用）。

    Args:
        config: 配置字典（含 RL 超参、奖励数值、预算等）
        seed: 世界种子；算法随机性由 config["algo_seed"]（缺省取 seed）驱动
        intrinsic_module: 内在奖励模块实例（ICM 或 RND）
    Returns:
        结果字典（格式与 run_single_episode 一致）
    """
    wrapper, db = make_env(config, seed)
    radius = int(config.get("search_radius", 10000))
    max_queries = int(config.get("max_queries_per_episode", 10000))
    target = int(config.get("target_deep_sea_points", 40))
    temp_enabled = bool(config.get("temperature_enabled", True))

    # --- RL 超参（配置驱动） ---
    lr = float(config.get("learning_rate", 0.1))
    gamma = float(config.get("discount_factor", 0.95))
    eps_start = float(config.get("epsilon_start", 1.0))
    eps_end = float(config.get("epsilon_end", 0.05))
    eps_decay = float(config.get("epsilon_decay", 0.995))
    num_directions = int(config.get("action_directions", 8))
    action_steps = list(config.get("action_steps", [1, 2, 4, 8]))
    num_actions = num_directions * len(action_steps)

    # --- 奖励数值（配置驱动） ---
    deep_reward = float(config.get("deep_sea_reward_value", 10.0))
    oob_penalty = float(config.get("out_of_bounds_penalty", -10.0))
    repeat_penalty = float(config.get("repeat_visit_penalty", -5.0))
    intrinsic_weight = float(config.get("intrinsic_weight", 1.0))

    algo_seed = int(config.get("algo_seed", seed))
    rng = random.Random((seed * 1000003 + algo_seed) & 0xFFFFFFFF)

    # --- 状态空间：坐标粗化（区块 // 4），不编码温度（与主框架最终架构一致） ---
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

        # --- 越界处理 ---
        if not within_radius(next_cx, next_cz, radius):
            reward = oob_penalty
            next_state = state_of(0, 0)
            target_chunk = (0, 0)
            done_move = True
        elif (next_cx, next_cz) in visited_chunks:
            # 重复访问：不消耗查询预算，仅惩罚
            reward = repeat_penalty
            target_chunk = (next_cx, next_cz)
            done_move = True
        else:
            visited_chunks.add((next_cx, next_cz))
            target_chunk = (next_cx, next_cz)
            done_move = False

        # --- 内在奖励（基于目标状态特征） ---
        if not done_move:
            cur_feat = encode_state_features(current[0], current[1], cur_temp)
            r_int = intrinsic_module.intrinsic(cur_feat, action, nxt_feat)
            intrinsic_module.train(cur_feat, action, nxt_feat)
            reward = intrinsic_weight * r_int
        # 越界/重复：保持纯惩罚，无内在奖励

        # --- 区块查询（非越界非重复时） ---
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

        # --- Q-learning 更新（非终止步，含 gamma 自举） ---
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
