"""
_baselines_common.py
基线算法公共工具（仅依赖 L0 基础设施层：cubiomes_wrapper / database）

需求.txt 第十一节 L4 关键约束：
    · 基线算法独立于 main_algorithm.py，不依赖 L2/L1，只依赖 L0
    · 每个算法有自己的独立执行循环
    · 返回格式与 main_algorithm.run_single_episode() 的结果字典保持一致，
      便于 scripts/run_baselines.py 统一聚合与绘图

本模块提供：
    · make_env()            构造环境（CubiomesWrapper + DeepSeaDatabase）
    · query_chunk_center()  对单个区块执行一次标准查询（温度探针 + 深海判定，
                            每区块 2 次黑盒查询，与主框架 center_point 微观策略口径一致）
    · build_result()        构造与 run_single_episode 同构的结果字典
    · MLP                   numpy 实现的轻量两层网络（供 ICM / RND 内在奖励使用）
"""

import numpy as np
from typing import List, Tuple, Optional

from ..cubiomes_wrapper import CubiomesWrapper
from ..database import DeepSeaDatabase

# 8 方向位移表，与主框架 _DIRECTION_DELTAS 保持一致
DIRECTION_DELTAS: List[Tuple[int, int]] = [
    (1, 0), (1, -1), (0, -1), (-1, -1),
    (-1, 0), (-1, 1), (0, 1), (1, 1),
]


def make_env(config: dict, seed: int) -> Tuple[CubiomesWrapper, DeepSeaDatabase]:
    """构造黑盒环境与数据库（基线只用 L0）"""
    wrapper = CubiomesWrapper(seed=seed)
    db = DeepSeaDatabase(config.get("deep_sea_distance_threshold", 416))
    return wrapper, db


def chunk_center(cx: int, cz: int) -> Tuple[int, int]:
    """区块中心世界坐标（与主框架 center_point 微观策略取点一致）"""
    return cx * 16 + 8, cz * 16 + 8


def within_radius(cx: int, cz: int, radius: int) -> bool:
    """区块是否在搜索范围内（半径 10000 格的正方形区域）"""
    return abs(cx * 16) <= radius and abs(cz * 16) <= radius


def query_chunk_center(wrapper: CubiomesWrapper, db: DeepSeaDatabase,
                       config: dict, cx: int, cz: int,
                       temperature_history: Optional[List] = None,
                       temperature_samples: Optional[List] = None,
                       temperature_enabled: bool = True):
    """
    对区块 (cx, cz) 执行标准查询：
        1. 查询区块中心温度（1 次查询，计入预算）
        2. 判定区块中心是否深海（1 次查询，计入预算）
        3. 若命中深海，写入数据库（含间距阈值去重），返回是否为新有效点

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
    构造与 main_algorithm.run_single_episode() 同构的结果字典，
    使 run_baselines.py 可以复用统一的聚合/绘图逻辑。
    """
    temperature_history = temperature_history or []
    temperature_samples = temperature_samples or []
    points = db.get_all_points()
    return {
        "seed": seed,
        "total_queries": wrapper.query_count,
        "deep_sea_points_found": len(db),
        "deep_sea_points": points,
        "final_stage": 1,          # 基线为单一算法，无三阶段概念，固定记为 1
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
    轻量两层全连接网络（numpy 实现，SGD 训练）。

    仅供 ICM（前向模型误差）与 RND（预测器 vs 固定目标网络）两类
    内在好奇心奖励计算使用，不依赖 torch/tf，保证零额外依赖。
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
        h = np.maximum(0.0, x @ self.W1 + self.b1)   # ReLU 隐层
        return h @ self.W2 + self.b2, h

    def predict(self, x) -> np.ndarray:
        """前向输出（不更新参数）"""
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        out, _ = self._forward(x)
        return out.reshape(-1)

    def train_step(self, x, y) -> float:
        """单步 SGD（MSE 损失），返回当前批次均方误差"""
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
    将区块坐标 + 温度编码为连续特征向量，供 ICM/RND 网络使用。
    scale 取搜索半径 10000 格对应的区块数 625，使坐标归一化到 [-1, 1] 附近。
    """
    t = 0.0 if temp is None else float(np.tanh(temp))
    return np.array([cx / scale, cz / scale, t], dtype=np.float64)


def make_action_onehot(action_idx: int, num_actions: int) -> np.ndarray:
    """动作索引 -> one-hot 向量（ICM 前向模型输入用）"""
    a = np.zeros(num_actions, dtype=np.float64)
    a[action_idx % num_actions] = 1.0
    return a
