
"""
main_algorithm.py
核心RL主骨架：协调宏观区块遍历、微观坐标选择、状态管理、奖励计算与Q-learning更新

【定位】需求第十五节第三阶段：本文件是"哑"的纯执行器。
    - 纯配置驱动：所有策略通过工厂模式按 yaml 字符串动态实例化
    - 无循环/无导出：只提供 run_single_episode() 返回结果字典，
      20次重复、CSV生成、图表绘制绝不在此文件中

【修复依据】(≤100字)
1. 奖励调用对齐 FullRewardBaseline.calculate 真实签名：原三重 try 兜底传入
   current_state/visited_states/current_chunk 等不被接受的参数，三次全部 TypeError，
   最终恒返回 0.0，Stage3 收不到任何梯度信号。
2. 温度注入微观选点 get_points(temperature=)：原调用漏传，adaptive_sampling 永久退化。
3. episode 重置时重建 finder/strategy 并调用 reset()：原内部游标跨 episode 累加，
   同一 Explorer 连续跑 400 次时第2次起不再从原点扫描。
4. 新增已访问区块去重：宏观遍历会重复生成同一区块，原实现反复查询白白耗尽预算。
5. 好奇心改用 StateManager.is_new_state/mark_visited 契约，移除重复的 visited_states。
6. 温度直方图分桶范围改为由实际观测数据计算，不再硬编码，符合需求第一节严禁先验。
"""

import random
import math
import sys
from typing import Tuple, Dict, List, Optional
from collections import deque, Counter

# ==================== 防御性导入 (解决 ImportError) ====================
try:
    from .cubiomes_wrapper import CubiomesWrapper
    from .database import DeepSeaDatabase
    from .state_manager import StateManager
    from .chunk_finders import (BaseFinder, RayScanFinder, SpiralScanFinder,
                               AdaptiveScanFinder, GridScanFinder, AnchorScanFinder)
    from .traversal_strategies import (
        BaseStrategy, CenterPoint, CornerPoints, RandomTwoPoints, Grid3x3,
        CornersAndCenter, AnyThreeCorners, TShape, AdaptiveSampling, DiagonalTwoPoints,
    )
    from .reward_modules import FullRewardBaseline
except ImportError:
    from src.cubiomes_wrapper import CubiomesWrapper
    from src.database import DeepSeaDatabase
    from src.state_manager import StateManager
    from src.chunk_finders import (BaseFinder, RayScanFinder, SpiralScanFinder,
                                   AdaptiveScanFinder, GridScanFinder, AnchorScanFinder)
    from src.traversal_strategies import (
        BaseStrategy, CenterPoint, CornerPoints, RandomTwoPoints, Grid3x3,
        CornersAndCenter, AnyThreeCorners, TShape, AdaptiveSampling, DiagonalTwoPoints,
    )
    from src.reward_modules import FullRewardBaseline


# ==================== 默认配置兜底 ====================
# 需求第十五节：保留本字典作为 yaml 缺失时的默认值。
# 键名与 configs/ 下 YAML 严格一致（需求第十节：配置文件驱动，严禁硬编码消融数值）。
STAGE1_BASELINE = {
    # --- 实验标识 (需求第十节：动态输出说明头需打印算法名) ---
    "experiment_name": "深海群系探索系统 - 黑盒学习框架",
    "output_mode": "simple",             # simple=简易版(20行平均值) / detailed=详细版(20次完整结果+实时查询输出)

    # --- 感知层 (Stage1 消融变量) ---
    # temperature_enabled: 温度总开关。False=纯盲搜(不查温度/不偏向/不进状态)。
    "temperature_enabled": True,
    # temperature_in_state: 温度是否进入 Q 状态编码（独立子开关，仅 temperature_enabled=True 时生效）。
    #   实测结论：温度对"是否深海"无定位力(海洋内 AUC=0.458)，进 Q 状态会膨胀状态空间、
    #   稀释学习(旧 temp_on 31.23 < temp_off 32.43)；但温度能精确预测海洋变体(Spearman=1.0)，
    #   且各温度带深海率非单调(frozen 48% > normal 24% > cold 18% > lukewarm 0%)，
    #   做宏观导向偏向有效(+0.79)。故最终架构：温度【不进】Q 状态(本键=False)，只做导向偏向。
    "temperature_in_state": False,
    "temperature_bins": 10,              # 仅用于温度直方图分桶数，范围由实测数据决定
    "temperature_precision": 2,          # 状态编码的温度量化精度(小数位)，非温度范围
    # 状态格空间粒度(区块数)。温度层分辨率为 16 区块，本值必须 >16 温度才有增量信息；
    # 若 <=16，温度可由区块坐标整除推出，temperature on/off 消融必然得到相同结果。
    "spatial_granularity": 32,
    # 微观选点默认 center_point：Stage1 实测最优(32.84)且查询开销最小(9138)。
    "micro_strategy": "center_point",

    # --- 温度导向搜索 (需求第一节第10条 + 第四节第27行：Stage3 基于已学习的温度规律驱动搜索) ---
    # 与 temperature_enabled(状态编码) 正交：本开关让 agent 用"自主统计的温度带→深海概率"
    # 偏向 Stage3 的宏观动作选择，温度不进奖励公式(守第6/30条)，仅 Stage3 生效(守第26行)。
    # 实测依据：frozen 带深海率 48.4%(1.77x基准)、lukewarm/warm 带≈0%，且关系非单调
    # (cold 17.9% < normal 23.5%)，故人工"越冷越好"启发式会失效，必须自主学习概率表。
    "temperature_guidance_enabled": True,
    "temperature_guidance_weight": 3.0,  # 探索先验权重(超参数，Stage4 可扫描)
    "temperature_guidance_min_samples": 20,  # 概率表生效所需最小样本数，不足则退回纯RL

    # --- 移动层 (Stage2 消融变量) ---
    "chunk_finder": "ray_scan",
    "deep_sea_distance_threshold": 416,

    # --- 奖励层 (Stage3 消融变量) ---
    "deep_sea_reward_enabled": True,
    "curiosity_reward_enabled": True,
    "penalty_reward_enabled": True,
    "deep_sea_reward_value": 10.0,
    "curiosity_reward_value": 3.0,
    "curiosity_weight": 1.0,
    "out_of_bounds_penalty": -10.0,
    "repeat_visit_penalty": -5.0,

    # --- RL超参数 ---
    "learning_rate": 0.1,
    "discount_factor": 0.95,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay": 0.995,
    "replay_buffer_size": 8192,          # 需求第一节：固定 8192
    "replay_batch_size": 16,
    "replay_frequency": 5,               # 每 N 步触发一次经验回放

    # --- 阶段切换阈值 (需求第四节) ---
    "stage2_min_deep_sea_points": 5,
    "stage3_min_deep_sea_points": 10,

    # --- 全局限制 (需求第一节第9条：具体数值由 Stage4 参数扫描确定) ---
    "max_queries_per_episode": 10000,
    "target_deep_sea_points": 40,
    "search_radius": 10000,

    # --- 动作空间 (需求第九节：32个动作 = 8方向 x 4步长) ---
    "action_directions": 8,
    "action_steps": [1, 2, 4, 8],
    "direction_memory_bonus": 0.5,       # 需求第一节：方向与步长记忆权重

    # --- Stage3 远跳机制（zhongxin 参考实现的 _get_start_from_deep_point 补全）---
    # 缺陷依据：Stage3 动作步长仅 16~128 格 << 深海点排斥区(deep_sea_distance_threshold)，
    # 局部移动的命中点全部挤在同一排斥区内被 database 去重，预算耗死在已发现点附近
    # （实测 Stage3 占 95%+ 预算、有效点率 0.74%）。远跳从已发现深海点向随机方向
    # 跳跃 threshold×[min_ratio, max_ratio] 格，跳出排斥区直达新片区；
    # 跳距全部由 deep_sea_distance_threshold 变量推导（消融扫阈值时自动适配，零硬编码）。
    "stage3_jump_enabled": True,         # 远跳开关
    "stage3_jump_interval": 30,          # Stage3 连续 N 步无新深海点后触发远跳
    "stage3_jump_min_ratio": 1.25,       # 跳距下限 = 间距阈值 × 本系数（>1 保证跳出排斥区）
    "stage3_jump_max_ratio": 5.0,        # 跳距上限 = 间距阈值 × 本系数
}


def _merge_config(user_config: dict, default_config: dict) -> dict:
    """递归合并用户配置与默认配置，用户配置优先"""
    merged = default_config.copy()
    for key, value in user_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_config(value, merged[key])
        else:
            merged[key] = value
    return merged


class ReplayBuffer:
    """固定容量的FIFO经验回放缓冲区（需求第一节：容量固定 8192）"""

    def __init__(self, capacity: int = 8192):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class QLearningAgent:
    """Q-learning智能体"""

    def __init__(self, config: dict):
        self.lr = config["learning_rate"]
        self.gamma = config["discount_factor"]
        self.epsilon_start = config["epsilon_start"]
        self.epsilon = self.epsilon_start
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_decay = config["epsilon_decay"]
        self.q_table: Dict[Tuple, List[float]] = {}

        # 动作空间由配置推导，严禁硬编码（需求第十节）
        self.num_directions = int(config["action_directions"])
        self.action_steps: List[int] = list(config["action_steps"])
        self.num_actions = self.num_directions * len(self.action_steps)

        # 需求第一节：记录曾带来深海发现的"方向+步长"组合，后续决策赋予更高权重
        self.direction_step_memory: Dict[Tuple[int, int], int] = {}
        self.memory_bonus = float(config.get("direction_memory_bonus", 0.5))

    def reset(self):
        """需求第一节第3条：无状态重置，清空Q-table与访问记录"""
        self.q_table.clear()
        self.direction_step_memory.clear()
        self.epsilon = self.epsilon_start

    def decode_indices(self, action: int) -> Tuple[int, int]:
        """动作索引 -> (方向索引, 步长索引)"""
        return action % self.num_directions, action // self.num_directions

    def record_discovery(self, action: int):
        key = self.decode_indices(action)
        self.direction_step_memory[key] = self.direction_step_memory.get(key, 0) + 1

    def get_q_values(self, state: Tuple) -> List[float]:
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.num_actions
        return self.q_table[state]

    def select_action(self, state: Tuple, is_stage3: bool = False) -> int:
        """epsilon-greedy 动作选择（含方向记忆 bonus）。行为与重构前完全一致。"""
        return self.select_action_with_bias(state, None, apply_memory=is_stage3)

    def select_action_with_bias(self, state: Tuple, action_bias, apply_memory: bool = True) -> int:
        """
        epsilon-greedy + 外部动作偏向（如温度导向探索先验）。

        action_bias: callable(action_idx) -> float，叠加在 Q 值与方向记忆 bonus 之上。
        偏向**不改变 epsilon 探索概率**，仅影响"利用"阶段的动作排序——
        因此它是探索先验，不是奖励，温度不进入任何奖励公式（守需求第6/30条）。
        平局时随机选择，保持与原 select_action 一致的随机性。
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        q_values = self.get_q_values(state).copy()
        if apply_memory and self.direction_step_memory:
            for action_idx in range(self.num_actions):
                key = self.decode_indices(action_idx)
                if key in self.direction_step_memory:
                    q_values[action_idx] += self.memory_bonus * self.direction_step_memory[key]

        if action_bias is not None:
            for action_idx in range(self.num_actions):
                q_values[action_idx] += action_bias(action_idx)

        max_q = max(q_values)
        best_actions = [i for i, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)
        target = reward if done else reward + self.gamma * max(next_q_values)
        q_values[action] += self.lr * (target - q_values[action])
        self.q_table[state] = q_values

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


class DeepSeaExplorer:
    """深海探索主类"""

    # 8方向位移表，顺序与 RayScanFinder._DIRECTIONS 一致：
    # 东、东南、南、西南、西、西北、北、东北
    _DIRECTION_DELTAS: List[Tuple[int, int]] = [
        (1, 0), (1, -1), (0, -1), (-1, -1),
        (-1, 0), (-1, 1), (0, 1), (1, 1),
    ]

    def __init__(self, config: dict):
        self.config = _merge_config(config or {}, STAGE1_BASELINE)
        self.cubiomes = CubiomesWrapper()

        # 内部状态字段必须先于工厂方法初始化：
        # _create_micro_strategy() 会读取 self._algo_seed 以保证随机策略可复现
        self._algo_seed: Optional[int] = None
        self._last_temperature_feedback: Optional[float] = None
        self._visited_chunks: set = set()
        self._stage_transitions: List[Dict[str, int]] = []
        # 温度探针次数：temp_on 与 temp_off 两组此值必然相等（预算公平性的验证依据）
        self._temperature_probe_count: int = 0
        # 温度格缓存 {(gx,gz): temp}：扫描区块时缓存温度，供 Stage3 温度偏向零额外查询
        self._temperature_grid_cache: Dict[Tuple[int, int], float] = {}
        # Stage3 远跳机制：连续无新深海点步数（达到 stage3_jump_interval 触发远跳）
        self._stage3_idle_steps: int = 0

        # 初始化数据库，传入配置的阈值（需求第三节）
        self.db = DeepSeaDatabase(self.config["deep_sea_distance_threshold"])

        # 温度进入 Q 状态编码，需总开关与子开关同时为真。
        # 子开关 temperature_in_state=False 时：温度仍查询、仍做导向偏向，但不进状态。
        self._temp_in_state = bool(
            self.config.get("temperature_enabled", True)
            and self.config.get("temperature_in_state", False)
        )
        self.state_manager = StateManager(
            temperature_enabled=self._temp_in_state,
            temperature_precision=self.config.get("temperature_precision", 2),
            spatial_granularity=self.config.get("spatial_granularity", 32),
        )
        # 配置合理性检查：仅当温度确实进状态时，空间粒度才须粗于温度分辨率(16区块)，
        # 否则温度可由区块坐标整除推出，对状态空间零贡献。
        if self._temp_in_state and not self.state_manager.granularity_is_coarse_enough:
            print(f"[警告] spatial_granularity={self.state_manager.spatial_granularity} "
                  f"未大于温度分辨率 {self.state_manager.temperature_resolution_chunks} 区块，"
                  f"温度可由区块坐标推出，对状态空间零贡献",
                  file=sys.stderr)
        self.finder = self._create_finder()
        self.micro_strategy = self._create_micro_strategy()
        self.reward_module = FullRewardBaseline(self.config)
        self.agent = QLearningAgent(self.config)
        self.replay_buffer = ReplayBuffer(self.config["replay_buffer_size"])

        self.deep_sea_points: List[Tuple[int, int, float]] = []
        self.current_chunk: Tuple[int, int] = (0, 0)
        self.current_state: Optional[Tuple] = None
        self.temperature_history: List[float] = []
        # 全区块温度样本 [(温度, 该区块是否含深海), ...]
        # 需求第一节第6条：只有同时持有深海与非深海样本，才能算出"温度→深海概率"，
        # AI 才能自主发现何种温度区间倾向深海。仅记录深海点温度无法形成概率。
        self.temperature_samples: List[Tuple[float, bool]] = []

        self._print_header()

    # ==================== 工厂方法（需求第十五节：字符串动态实例化） ====================

    def _create_finder(self) -> BaseFinder:
        finder_map = {
            "ray_scan": RayScanFinder,
            "spiral_scan": SpiralScanFinder,
            "adaptive_scan": AdaptiveScanFinder,
            "grid_scan": GridScanFinder,
            "anchor_scan": AnchorScanFinder,
        }
        name = self.config["chunk_finder"]
        # 未知策略名必须显式报错：静默退回 ray_scan 会让消融实验在错误配置下"正常"跑完，
        # 产出看似合理实则张冠李戴的结果，且无任何提示（配置驱动的底线要求）。
        if name not in finder_map:
            raise KeyError(
                f"未知宏观遍历策略 '{name}'，可用值: {sorted(finder_map.keys())}"
            )
        return finder_map[name](self.config)

    def _create_micro_strategy(self) -> BaseStrategy:
        strategy_map = {
            "center_point": CenterPoint,
            "corner_points": CornerPoints,
            "random_two_points": RandomTwoPoints,
            "grid_3x3": Grid3x3,
            "corners_and_center": CornersAndCenter,
            "any_three_corners": AnyThreeCorners,
            "t_shape": TShape,
            "adaptive_sampling": AdaptiveSampling,
            "diagonal_two_points": DiagonalTwoPoints,
        }
        name = self.config["micro_strategy"]
        strategy_class = strategy_map.get(name, CornersAndCenter)
        # 注入算法种子，使含随机性的策略可复现（需求第十二节：分离环境随机性与算法随机性）
        kwargs = dict(self.config)
        kwargs["seed"] = self._algo_seed
        return strategy_class(**kwargs)

    # ==================== 输出说明头（需求第十节：动态输出说明头） ====================

    @property
    def is_detailed_mode(self) -> bool:
        """需求第十节：详细版输出20次完整结果+实时查询输出；简易版仅输出平均值"""
        return str(self.config.get("output_mode", "simple")).lower() == "detailed"

    def _reward_switch_states(self) -> str:
        """
        4个奖励开关状态（需求第五节：奖励模块拆分为4个独立文件）
        深海发现 / 好奇心 / 惩罚 三项子开关 + 全开基线聚合器状态
        """
        rm = self.reward_module
        deep = "开" if rm.deep_sea.enabled else "关"
        curi = "开" if rm.curiosity.enabled else "关"
        pen = "开" if rm.penalty.enabled else "关"
        # 全开基线：三项子奖励全部启用时才是"全开对照组"
        all_on = "全开对照组" if (rm.deep_sea.enabled and rm.curiosity.enabled and rm.penalty.enabled) else "部分关闭"
        return (f"深海发现={deep} | 好奇心={curi} | 惩罚={pen} | 全奖励基线={all_on}")

    def _print_header(self):
        """
        需求第十节：程序启动时必须读取当前 yaml，动态生成并打印表头，
        包含算法名、间距、采样、遍历、4个奖励开关状态、温度开关、查询上限等信息。
        """
        cfg = self.config
        rm = self.reward_module
        temp_status = "ON (黑盒触觉)" if cfg["temperature_enabled"] else "OFF (纯盲搜)"

        print("\n" + "=" * 74)
        print(f"  算法名        : {cfg.get('experiment_name')}")
        print(f"  输出模式      : {'详细版 (20次完整结果 + 实时查询输出)' if self.is_detailed_mode else '简易版 (20行平均值)'}")
        print("-" * 74)
        print(f"  宏观遍历策略  : {cfg['chunk_finder']}")
        print(f"  微观采样策略  : {cfg['micro_strategy']}")
        print(f"  温度感知开关  : {temp_status}")
        # 温度导向搜索（需求第四节第27行）：仅 Stage3 生效，依赖温度感知开启
        if not cfg["temperature_enabled"]:
            guidance_status = "OFF (温度感知关闭，偏向自动失效)"
        elif cfg.get("temperature_guidance_enabled", False):
            guidance_status = (f"ON (Stage3 概率偏向, 权重 {cfg.get('temperature_guidance_weight')}, "
                               f"最小样本 {cfg.get('temperature_guidance_min_samples')})")
        else:
            guidance_status = "OFF"
        print(f"  温度导向搜索  : {guidance_status}")
        # 温度是否进入 Q 状态编码（与导向偏向正交）。实测：进状态有害、做偏向有益。
        if not cfg["temperature_enabled"]:
            state_temp_status = "OFF (温度感知关闭)"
        elif self._temp_in_state:
            state_temp_status = "ON (温度量化后参与状态编码)"
        else:
            state_temp_status = "OFF (温度不进Q状态，仅做导向偏向 —— 最终架构)"
        print(f"  温度进状态编码: {state_temp_status}")
        print(f"  深海点间距阈值: {cfg['deep_sea_distance_threshold']} 格")
        # 状态编码配置：仅当温度进状态时，粒度才须粗于温度分辨率(16区块)
        gran = self.state_manager.spatial_granularity
        prec = self.state_manager.temperature_precision
        if self._temp_in_state:
            gran_note = ("有效: 粗于温度分辨率16区块"
                         if self.state_manager.granularity_is_coarse_enough
                         else "⚠ 无效: 未粗于温度分辨率16区块")
        else:
            gran_note = "温度不进状态，粒度仅决定坐标区域划分"
        print(f"  状态空间粒度  : {gran} 区块/格 ({gran_note})")
        print(f"  温度量化精度  : 保留 {prec} 位小数 (分辨率参数, 非温度范围)")
        print("-" * 74)
        print(f"  4个奖励开关状态:")
        print(f"    {self._reward_switch_states()}")
        print(f"    深海发现奖励值 : +{rm.deep_sea.reward}")
        print(f"    好奇心奖励值   : +{rm.curiosity.reward} (权重 {rm.curiosity.weight})")
        print(f"    越界惩罚       : {rm.penalty.out_of_bounds}")
        print(f"    重复访问惩罚   : {rm.penalty.repeated_visit}")
        print("-" * 74)
        print(f"  查询上限      : {cfg['max_queries_per_episode']} 次 / 目标 {cfg['target_deep_sea_points']} 个深海点")
        print(f"  搜索半径      : {cfg['search_radius']} 格 (正方形区域, 仅用 X/Z)")
        print(f"  经验回放缓冲区: {cfg['replay_buffer_size']}")
        print(f"  阶段门限      : Stage2 ≥ {cfg['stage2_min_deep_sea_points']} 点, "
              f"Stage3 ≥ {cfg['stage3_min_deep_sea_points']} 点")
        print(f"  动作空间      : {cfg['action_directions']} 方向 x {len(cfg['action_steps'])} 步长 "
              f"= {self.agent.num_actions} 个动作, 步长={cfg['action_steps']}")
        print("=" * 74 + "\n")

    # ==================== 状态重置（需求第一节：无状态重置） ====================

    def _reset_state(self):
        """每次实验开始前清空所有状态，确保无历史经验残留"""
        self.agent.reset()
        self.replay_buffer = ReplayBuffer(self.config["replay_buffer_size"])
        self.deep_sea_points.clear()
        self.temperature_history.clear()
        self.temperature_samples.clear()
        self._visited_chunks.clear()
        self._stage_transitions.clear()
        self._last_temperature_feedback = None
        self._temperature_probe_count = 0
        self._temperature_grid_cache.clear()
        self._stage3_idle_steps = 0
        self.state_manager.reset()
        self.db.clear()

        # 黑盒查询计数归零，作为查询预算的唯一计量口径
        self.cubiomes.query_count = 0

        # 【关键】重建 finder 与 micro_strategy 并重置其内部游标。
        # RayScanFinder._step / SpiralScanFinder._x,_z 等若不清零，
        # 同一 Explorer 连续跑多个 episode 时会从上次终点继续外扩，造成 episode 互相污染。
        self.finder = self._create_finder()
        self.micro_strategy = self._create_micro_strategy()
        self.finder.reset()

        self.current_chunk = (0, 0)
        self.current_state = self.state_manager.get_state(0, 0, temperature=None)
        self.state_manager.mark_visited(self.current_state)

    # ==================== 阶段与动作 ====================

    def _get_current_stage(self) -> int:
        """需求第四节：三阶段学习机制"""
        num_points = len(self.deep_sea_points)
        if num_points >= self.config["stage3_min_deep_sea_points"]:
            return 3
        elif num_points >= self.config["stage2_min_deep_sea_points"]:
            return 2
        return 1

    def _decode_action(self, action: int) -> Tuple[int, int, int, int, int]:
        """动作索引 -> (方向索引, 步长索引, 步长值, dx, dz)"""
        dir_idx, step_idx = self.agent.decode_indices(action)
        step = self.config["action_steps"][step_idx]
        dx, dz = self._DIRECTION_DELTAS[dir_idx]
        return dir_idx, step_idx, step, dx, dz

    def _is_out_of_bounds(self, x: int, z: int) -> bool:
        """需求第二节：限定在以出生点为中心、半径 search_radius 格的正方形区域内"""
        radius = self.config["search_radius"]
        return abs(x) > radius or abs(z) > radius

    # ==================== 反馈注入 ====================

    def _sync_finder_density(self):
        """需求第一节第8条：向宏观策略注入深海点密度，实现自适应搜索粒度"""
        updater = getattr(self.finder, "update_deep_sea_density", None)
        if updater is None:
            return
        chunks = [(x // 16, z // 16) for x, z, _ in self.deep_sea_points]
        updater(chunks)

    def _sync_strategy_reference_temps(self):
        """需求第九节：adaptive_sampling 需接收温度反馈作为参数"""
        setter = getattr(self.micro_strategy, "set_reference_temps", None)
        if setter is None:
            return
        temps = [t for _, _, t in self.deep_sea_points if t is not None]
        setter(temps)

    # ==================== 温度探针（保证消融预算公平） ====================

    def _probe_temperature(self, x: int, z: int) -> Optional[float]:
        """
        统一的温度查询入口。**无论 temperature_enabled 为何值都执行查询并计入预算**，
        仅在开关关闭时丢弃返回值（返回 None，即温度为"无"）。

        为什么必须这样设计：
            若关闭温度时跳过查询，temp_off 组每区块会比 temp_on 组少消耗 1 次查询，
            相同预算下能多扫约 22% 的区块（corners_and_center 实测）。
            于是两组差异会同时包含"温度特征的作用"与"扫描区块数的差异"，
            无法归因，Stage1 的核心消融即失去控制变量的意义。
            让两组付出完全相同的查询代价，唯一差异才是"温度是否参与状态编码"。

        Returns:
            temperature_enabled=True  -> 实际温度浮点值（查询失败为 None）
            temperature_enabled=False -> 恒为 None（查询已执行并计费，结果被丢弃）
        """
        if self.cubiomes.query_count >= self.config["max_queries_per_episode"]:
            return None

        try:
            temp = self.cubiomes.get_temperature(x, z)
        except Exception as e:
            print(f"[警告] get_temperature({x},{z}) 异常: {e}", file=sys.stderr)
            return None

        self._temperature_probe_count += 1
        # 温度消融关闭：结果不入状态、不入统计、不注入策略，温度为"无"
        if not self.config["temperature_enabled"]:
            return None
        return temp

    # ==================== 温度统计（需求第一节第6条：自主温度分析） ====================

    def _valid_temperatures(self) -> List[float]:
        """深海点温度样本（过滤缺失与 NaN）"""
        return [t for t in self.temperature_history
                if t is not None and not (isinstance(t, float) and math.isnan(t))]

    def _valid_temperature_samples(self) -> List[Tuple[float, bool]]:
        """全区块温度样本 [(温度, 是否深海)]，过滤缺失与 NaN"""
        return [(t, d) for t, d in self.temperature_samples
                if t is not None and not (isinstance(t, float) and math.isnan(t))]

    def _build_temperature_bins(self) -> List[Tuple[str, float, float]]:
        """
        按本 episode 实际观测到的温度范围 (min~max) 划分等宽分桶。
        不硬编码任何温度区间 —— 符合需求第一节「严禁先验知识」与
        第6条「自主识别…无需人工设定阈值」。
        Returns: [(桶标签, 下界, 上界), ...]
        """
        samples = self._valid_temperature_samples()
        if not samples:
            return []

        bins = max(int(self.config.get("temperature_bins", 10)), 1)
        lo = min(t for t, _ in samples)
        hi = max(t for t, _ in samples)
        if hi <= lo:
            # 全部样本温度相同，单桶
            return [(f"{lo:.2f}~{lo:.2f}", lo, hi)]

        width = (hi - lo) / bins
        edges = []
        for i in range(bins):
            b_lo = lo + i * width
            b_hi = lo + (i + 1) * width
            edges.append((f"{b_lo:.2f}~{b_hi:.2f}", b_lo, b_hi))
        return edges

    def _bin_index(self, temp: float, edges: List[Tuple[str, float, float]]) -> int:
        """把温度值映射到分桶索引（末桶闭区间，含上界）"""
        n = len(edges)
        for i in range(n - 1):
            if temp < edges[i][2]:
                return i
        return n - 1

    def _build_temperature_histogram(self) -> Dict[str, int]:
        """
        温度分布直方图：每个桶内的样本总数（含深海与非深海）。
        保留原键名与语义，供结果字典向后兼容。
        """
        edges = self._build_temperature_bins()
        if not edges:
            return {}

        histogram = Counter()
        for temp, _ in self._valid_temperature_samples():
            histogram[edges[self._bin_index(temp, edges)][0]] += 1
        return {label: histogram.get(label, 0) for label, _, _ in edges}

    def build_temperature_deep_probability(self) -> Dict[str, Dict[str, object]]:
        """
        需求第一节第6条：自动绘制「温度 → 深海概率」直方图数据，
        自主识别深海倾向的温度区间，无需人工设定阈值。

        概率 = 该温度桶内的深海区块数 / 该桶内的全部区块数
        （分母由"每个区块都查温度"提供，这正是必须全区块采样温度的原因）

        Returns:
            {桶标签: {"deep": 深海数, "non_deep": 非深海数, "total": 总数,
                      "probability": 深海概率}}
        """
        edges = self._build_temperature_bins()
        if not edges:
            return {}

        deep_counts = [0] * len(edges)
        total_counts = [0] * len(edges)
        for temp, is_deep in self._valid_temperature_samples():
            idx = self._bin_index(temp, edges)
            total_counts[idx] += 1
            if is_deep:
                deep_counts[idx] += 1

        table = {}
        for i, (label, _, _) in enumerate(edges):
            total = total_counts[i]
            table[label] = {
                "deep": deep_counts[i],
                "non_deep": total - deep_counts[i],
                "total": total,
                "probability": (deep_counts[i] / total) if total > 0 else 0.0,
            }
        return table

    def _analyze_temperature_preference(self) -> Dict[str, object]:
        """
        自主识别深海倾向的温度区间（需求第一节第6条）。
        仅基于已积累数据判定概率最高的桶，不引入任何人工先验阈值。
        """
        prob_table = self.build_temperature_deep_probability()
        samples = self._valid_temperature_samples()
        if not prob_table or not samples:
            return {}

        temps = [t for t, _ in samples]
        deep_temps = [t for t, d in samples if d]
        non_deep_temps = [t for t, d in samples if not d]

        # 概率最高的桶 = AI 自主发现的深海倾向温度区间。
        # 要求桶内至少有一定样本量，避免小样本桶因偶然命中而概率虚高。
        min_samples = max(int(len(samples) * 0.02), 3)
        candidates = {k: v for k, v in prob_table.items() if v["total"] >= min_samples}
        search_space = candidates if candidates else prob_table
        best_label, best_info = max(search_space.items(), key=lambda kv: kv[1]["probability"])

        analysis = {
            "sample_count": len(samples),
            "deep_sample_count": len(deep_temps),
            "non_deep_sample_count": len(non_deep_temps),
            "mean_temperature": sum(temps) / len(temps),
            "min_temperature": min(temps),
            "max_temperature": max(temps),
            "best_temperature_bin": best_label,
            "best_bin_probability": best_info["probability"],
            "best_bin_deep": best_info["deep"],
            "best_bin_total": best_info["total"],
            "temperature_histogram": self._build_temperature_histogram(),
            "temperature_deep_probability": prob_table,
        }

        # 深海与非深海的温度均值对比，用于判断温度特征是否具备区分度
        if deep_temps and non_deep_temps:
            deep_mean = sum(deep_temps) / len(deep_temps)
            non_deep_mean = sum(non_deep_temps) / len(non_deep_temps)
            analysis["deep_mean_temperature"] = deep_mean
            analysis["non_deep_mean_temperature"] = non_deep_mean
            analysis["temperature_gap"] = abs(deep_mean - non_deep_mean)

        return analysis

    def print_temperature_deep_probability(self) -> None:
        """
        打印「温度 → 深海概率」直方图（需求第一节第6条：AI 自主发现，无需人工设定阈值）
        仅在详细版输出模式下调用。
        """
        prob_table = self.build_temperature_deep_probability()
        if not prob_table:
            print("  温度样本不足，无法绘制温度→深海概率直方图。")
            return

        total_samples = sum(v["total"] for v in prob_table.values())
        total_deep = sum(v["deep"] for v in prob_table.values())

        print(f"\n  === 温度 → 深海概率直方图（AI 自主分析，分桶范围取自实测数据） ===")
        print(f"  {'温度区间':<16} {'深海数':>7} {'非深海数':>9} {'深海概率':>9}  分布")
        print(f"  {'─' * 62}")
        for label, info in prob_table.items():
            bar = '#' * int(info["probability"] * 30)
            print(f"  {label:<16} {info['deep']:>7} {info['non_deep']:>9} "
                  f"{info['probability']:>8.2%}  {bar}")
        print(f"  {'─' * 62}")
        if total_samples:
            print(f"  样本合计: {total_samples} 个区块 | 深海区块 {total_deep} 个 "
                  f"({total_deep / total_samples:.2%})")
        else:
            print("  样本合计: 0")

        best_label = max(prob_table.items(), key=lambda kv: kv[1]["probability"])
        if best_label[1]["probability"] > 0:
            print(f"  [AI 发现] 深海概率最高的温度区间为 {best_label[0]}，"
                  f"概率 {best_label[1]['probability']:.2%}")

    # ============ 温度导向搜索（需求第一节第10条 + 第四节第27行）============

    @staticmethod
    def _wilson_lower_bound(deep: int, total: int, z: float = 1.96) -> float:
        """
        Wilson score 置信区间下界（95%）。
        估计每个温度带的深海概率下界：小样本带自动压低，避免偶然命中导致概率虚高，
        比点估计 deep/total 更稳健（实测中 3 样本桶曾虚报到 66.7%）。

        零人工阈值：z=1.96 是 95% 置信水平的统计常量，与"何种温度更接近深海"无关，
        不构成需求第一节禁止的先验知识。
        """
        if total <= 0:
            return 0.0
        n = float(total)
        p = deep / n
        denom = 1.0 + z * z / n
        center = p + z * z / (2.0 * n)
        margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
        return max(0.0, (center - margin) / denom)

    def _build_temperature_band_lower_bounds(self):
        """
        构建「温度带 → 深海概率 Wilson 下界」表与基准率。
        分桶完全数据驱动（范围取自本 episode 实测温度 min~max，复用 _build_temperature_bins），
        不使用 frozen/cold/warm 等人工温度阈值（守需求第一节"严禁先验知识"）。

        Returns:
            (band_lower_bounds: {桶索引: Wilson下界}, base_rate: 整体深海率)
            样本不足时返回 ({}, 0.0)。
        """
        edges = self._build_temperature_bins()
        if not edges:
            return {}, 0.0

        deep_counts = [0] * len(edges)
        total_counts = [0] * len(edges)
        for temp, is_deep in self._valid_temperature_samples():
            idx = self._bin_index(temp, edges)
            total_counts[idx] += 1
            if is_deep:
                deep_counts[idx] += 1

        total_all = sum(total_counts)
        if total_all <= 0:
            return {}, 0.0
        base_rate = sum(deep_counts) / total_all

        lower_bounds = {
            i: self._wilson_lower_bound(deep_counts[i], total_counts[i])
            for i in range(len(edges)) if total_counts[i] > 0
        }
        return lower_bounds, base_rate

    def _temperature_grid_key(self, chunk_x: int, chunk_z: int) -> Tuple[int, int]:
        """
        温度格坐标：温度层分辨率 16 区块(256方块)，同格内温度恒定。
        源码依据 cubiomes generator.h:57 (L_OCEAN_TEMP_256)、layers.c:3130 (x/8.0)。
        用于缓存已查区块温度，使动作偏向零额外查询开销。
        """
        return chunk_x // 16, chunk_z // 16

    def _make_temperature_action_bias(self):
        """
        构建 Stage3 温度导向动作偏向函数（需求第四节第27行：基于已学习的温度规律驱动搜索）。

        偏向量 = weight × (目的区块温度带的 Wilson 下界 − 基准率)
          高概率带(实测 frozen 1.77x) → 正偏向 → 倾向移入并停留精搜
          低概率带(实测 lukewarm/warm≈0) → 负偏向 → 倾向迁离
          未知温度格/未知带 → 0（中性，退回纯 RL）

        合规三红线：
          1 零人工阈值：温度带数据驱动分桶，概率用 Wilson 下界，无人工温度区间。
          2 不进奖励公式：偏向只叠加在动作选择的 Q 值上（探索先验），奖励仍排除温度（守第6/30条）。
          3 仅 Stage3 生效：Stage1 盲搜、Stage2 仅收集不决策（守第25/26行）。

        零额外查询：目的区块温度取自 _temperature_grid_cache（扫描时已缓存），
        未缓存格按中性处理，绝不为偏向额外查询温度。

        Returns:
            callable(action_idx)->float，或 None（功能关闭/温度关闭/样本不足）。
        """
        if not self.config.get("temperature_guidance_enabled", False):
            return None
        if not self.config.get("temperature_enabled", True):
            return None

        min_samples = int(self.config.get("temperature_guidance_min_samples", 20))
        if len(self._valid_temperature_samples()) < min_samples:
            return None

        lower_bounds, base_rate = self._build_temperature_band_lower_bounds()
        if not lower_bounds or base_rate <= 0:
            return None

        edges = self._build_temperature_bins()
        weight = float(self.config.get("temperature_guidance_weight", 5.0))
        origin_chunk = self.current_chunk
        cache = self._temperature_grid_cache

        def bias(action_idx: int) -> float:
            _, _, step, dx, dz = self._decode_action(action_idx)
            target = (origin_chunk[0] + dx * step, origin_chunk[1] + dz * step)
            temp = cache.get(self._temperature_grid_key(*target))
            if temp is None:
                return 0.0
            lb = lower_bounds.get(self._bin_index(temp, edges))
            if lb is None:
                return 0.0
            return weight * (lb - base_rate)

        return bias

    # ==================== Stage3 远跳机制（需求参考实现 zhongxin.py 的
    #  _get_start_from_deep_point 补全：从深海点向外远跳，跳出 416 格排斥区） ====================

    def _stage3_far_jump(self) -> bool:
        """
        Stage3 无进展时从已发现深海点向随机方向远距离跳跃。

        为什么需要远跳（实测根因）：
            Stage3 动作步长 16~128 格，而深海点排斥区 = deep_sea_distance_threshold 格。
            纯局部移动的命中点全部落在已有点排斥区内被 database 去重，有效点率仅 0.74%，
            预算耗死在已发现点附近。远跳让 agent 跳出排斥区直达新片区。

        跳距完全由 deep_sea_distance_threshold 变量推导（消融扫阈值时自动适配）：
            min_dist = threshold × stage3_jump_min_ratio（>1，保证跳出排斥区）
            max_dist = threshold × stage3_jump_max_ratio
        与 zhongxin.py 的 [512, 1024] 格（其排斥区 512 格 → 1~2 倍阈值）同一量级。

        目标区块需在搜索半径内且未访问；重试 16 次失败后收缩到阈值附近再试，
        仍失败返回 False（本轮退回普通动作）。

        Returns:
            True=已跳转到新目标区块；False=未跳转
        """
        threshold = float(self.config["deep_sea_distance_threshold"])
        min_ratio = float(self.config.get("stage3_jump_min_ratio", 1.25))
        max_ratio = float(self.config.get("stage3_jump_max_ratio", 5.0))
        min_dist = max(int(threshold * min_ratio), int(threshold) + 32)  # 格
        max_dist = int(threshold * max_ratio)
        if max_dist <= min_dist:
            max_dist = min_dist + 64

        # 第一轮：阈值×[min_ratio, max_ratio] 的宽域跳跃
        for _ in range(16):
            px, pz, _ = random.choice(self.deep_sea_points)
            dx, dz = random.choice(self._DIRECTION_DELTAS)
            dist = random.randint(min_dist, max_dist)
            nx = px // 16 + dx * (dist // 16)      # 深海点(格) -> 区块 + 区块位移
            nz = pz // 16 + dz * (dist // 16)
            if self._is_out_of_bounds(nx * 16, nz * 16):
                continue
            if (nx, nz) in self._visited_chunks:
                continue
            self.current_chunk = (nx, nz)
            return True

        # 第二轮：收缩到阈值附近，提高边界内命中率
        for _ in range(16):
            px, pz, _ = random.choice(self.deep_sea_points)
            dx, dz = random.choice(self._DIRECTION_DELTAS)
            dist = random.randint(min_dist, min_dist + 128)
            nx = px // 16 + dx * (dist // 16)
            nz = pz // 16 + dz * (dist // 16)
            if self._is_out_of_bounds(nx * 16, nz * 16):
                continue
            if (nx, nz) in self._visited_chunks:
                continue
            self.current_chunk = (nx, nz)
            return True
        return False

    # ==================== 学习更新 ====================

    def _replay_batch_update(self):
        """需求第一节：经验回放——从缓冲区采样批量重放，打破样本相关性"""
        batch_size = int(self.config.get("replay_batch_size", 16))
        if batch_size <= 0 or len(self.replay_buffer) < batch_size:
            return
        for s, a, r, ns, d in self.replay_buffer.sample(batch_size):
            self.agent.update(s, a, r, ns, d)

    def _learn(self, action: int, reward: float, next_state: Tuple,
               done: bool = False, step_index: int = 0):
        """单步 Q-learning 更新 + 经验入池 + 按频率批量重放 + epsilon 衰减"""
        self.agent.update(self.current_state, action, reward, next_state, done)
        self.replay_buffer.push(self.current_state, action, reward, next_state, done)

        frequency = int(self.config.get("replay_frequency", 5))
        if frequency > 0 and step_index % frequency == 0:
            self._replay_batch_update()

        self.agent.decay_epsilon()

    def _calculate_reward(self, is_new_deep_sea: bool, distance_to_nearest: float,
                          is_new_state: bool, is_out_of_bounds: bool = False,
                          is_repeated_visit: bool = False) -> float:
        """
        按 FullRewardBaseline.calculate 的真实签名调用。
        需求第五节：总奖励为深海发现 + 好奇心 + 惩罚三项之和；温度不参与奖励计算。
        """
        try:
            return self.reward_module.calculate(
                is_new_deep_sea=is_new_deep_sea,
                distance_to_nearest=distance_to_nearest,
                is_new_state=is_new_state,
                is_out_of_bounds=is_out_of_bounds,
                is_repeated_visit=is_repeated_visit,
            )
        except Exception as e:
            print(f"[警告] 奖励计算失败: {e}，本步奖励记为0", file=sys.stderr)
            return 0.0

    # ==================== 单次 Episode ====================

    def run_single_episode(self, seed: int) -> dict:
        """
        执行单次完整搜索 episode。
        需求第十五节：只提供本方法返回结果字典，20次重复/CSV/图表绝不在此文件中。
        """
        self._algo_seed = seed
        self._reset_state()
        random.seed(seed)

        stage = 1
        max_queries = self.config["max_queries_per_episode"]
        target_points = self.config["target_deep_sea_points"]
        convergence_data = []
        last_experience = None
        step_index = 0

        # 查询预算以黑盒实际调用次数为准（is_deep_ocean 与 get_temperature 均计费）
        while self.cubiomes.query_count < max_queries and len(self.deep_sea_points) < target_points:
            stage = self._get_current_stage()
            action_taken = 0
            is_new_valid_point = False
            nearest_dist = 0.0
            found_temperature = None
            step_index += 1

            # === 宏观：获取下一个区块 ===
            if stage in (1, 2):
                # 需求第四节：Stage1 盲搜、Stage2 仅收集统计不做决策，均按预设步进策略扫描
                cx, cz = self.current_chunk
                self.current_chunk = self.finder.get_next_chunk(cx, cz)
            else:
                # 需求第四节：Stage3 启动奖励算法驱动高效搜索
                # 需求第四节第27行：Stage3 基于已学习的温度规律驱动高效搜索。
                # 温度偏向为探索先验，叠加在动作选择的 Q 值上，不进奖励公式（守第6/30条）。
                # 样本不足/功能关闭时 _make_temperature_action_bias 返回 None，自动退回纯 RL。
                self._stage3_idle_steps += 1
                # === 远跳机制：连续 N 步无新深海点 → 从深海点远跳跳出排斥区 ===
                # 不改变动作空间/奖励/Q-learning，仅作为探索重启（参考实现补全）。
                if (self.config.get("stage3_jump_enabled", True)
                        and self._stage3_idle_steps >= self.config.get("stage3_jump_interval", 30)
                        and self.deep_sea_points
                        and self._stage3_far_jump()):
                    self._stage3_idle_steps = 0
                    # 远跳后已更新 current_chunk，直接进入本轮查询流程
                else:
                    temp_bias = self._make_temperature_action_bias()
                    action_taken = self.agent.select_action_with_bias(
                        self.current_state, temp_bias, apply_memory=True
                    )
                    _, _, step, dx, dz = self._decode_action(action_taken)
                    cx, cz = self.current_chunk
                    self.current_chunk = (cx + dx * step, cz + dz * step)

            chunk_x, chunk_z = self.current_chunk

            # === 区块去重：跳过已扫描区块，避免重复消耗查询预算 ===
            # 宏观遍历会重复生成同一区块坐标，若不去重则同一批区块被反复查询。
            if (chunk_x, chunk_z) in self._visited_chunks:
                next_state = self.state_manager.get_state(chunk_x, chunk_z, temperature=None)
                if stage == 3:
                    # 需求第五节：重复访问惩罚
                    reward = self._calculate_reward(
                        is_new_deep_sea=False,
                        distance_to_nearest=0.0,
                        is_new_state=self.state_manager.is_new_state(next_state),
                        is_repeated_visit=True,
                    )
                    self._learn(action_taken, reward, next_state, step_index=step_index)
                    last_experience = (self.current_state, action_taken, reward, next_state, False)
                self.state_manager.mark_visited(next_state)
                self.current_state = next_state
                continue

            self._visited_chunks.add((chunk_x, chunk_z))

            # === 越界检查（需求第二节）===
            if self._is_out_of_bounds(chunk_x * 16, chunk_z * 16):
                next_state = self.state_manager.get_state(0, 0, temperature=None)
                if stage == 3:
                    # 惩罚统一由 PenaltyReward 计算，不在此处硬编码数值（需求第十节）
                    reward = self._calculate_reward(
                        is_new_deep_sea=False,
                        distance_to_nearest=0.0,
                        is_new_state=self.state_manager.is_new_state(next_state),
                        is_out_of_bounds=True,
                    )
                    self._learn(action_taken, reward, next_state, step_index=step_index)
                    last_experience = (self.current_state, action_taken, reward, next_state, False)

                self.current_chunk = (0, 0)
                self.state_manager.mark_visited(next_state)
                self.current_state = next_state
                convergence_data.append({
                    "query_count": self.cubiomes.query_count,
                    "deep_sea_points": len(self.deep_sea_points),
                    "stage": stage,
                })
                continue

            # === 全区块温度查询（需求第一节第6条：AI 自主发现温度与深海的关系）===
            # 每个区块都查一次中心点温度，无论是否深海都记录 (温度, 是否含深海) 样本对。
            # 只有同时持有深海与非深海样本，才能统计出"温度 → 深海概率"，
            # 机器才能自主学出哪种温度区间倾向深海，而不是只见过深海点的温度。
            # 经 _probe_temperature 统一计费：温度开关关闭时仍查询但丢弃结果，
            # 保证 temp_on/temp_off 两组扫描的区块数严格相同（消融预算公平，见该方法文档）。
            center_x, center_z = chunk_x * 16 + 8, chunk_z * 16 + 8
            chunk_temperature = self._probe_temperature(center_x, center_z)

            # 写入温度格缓存：供 Stage3 温度偏向零额外查询使用（温度关闭时为 None，不入缓存）
            if chunk_temperature is not None:
                self._temperature_grid_cache[
                    self._temperature_grid_key(chunk_x, chunk_z)
                ] = chunk_temperature

            # 当前区块温度即为注入微观策略的温度反馈（adaptive_sampling 依赖此值动态调整）
            # 温度关闭时 chunk_temperature 恒为 None，微观策略收到"无温度"，自动退化为几何采样
            self._last_temperature_feedback = chunk_temperature

            # === 微观：区块内选点（注入当前区块温度反馈）===
            sample_points = self.micro_strategy.get_points(
                chunk_x, chunk_z, temperature=chunk_temperature
            )

            # === 查询每个采样点 ===
            chunk_has_deep = False
            for wx, wz in sample_points:
                if self.cubiomes.query_count >= max_queries:
                    break

                try:
                    is_deep = self.cubiomes.is_deep_ocean(wx, wz)
                except Exception as e:
                    print(f"[警告] is_deep_ocean({wx},{wz}) 异常: {e}", file=sys.stderr)
                    continue

                if is_deep:
                    chunk_has_deep = True
                    # 需求第三节：查询到深海点时同步获取该点温度值。
                    # 复用判断仅按坐标重合（不以 chunk_temperature 是否为 None 为条件）：
                    # 温度关闭时区块中心已被 _probe_temperature 计费过，此处若因 temp 为 None
                    # 而不复用，temp_off 组就会比 temp_on 组多查一次，破坏消融预算公平。
                    if (wx, wz) == (center_x, center_z):
                        temp = chunk_temperature          # 已在区块中心查过，直接复用（关闭时为 None）
                    else:
                        temp = self._probe_temperature(wx, wz)   # 非中心点，两组都计费查询

                    if self.config["temperature_enabled"]:
                        self.temperature_history.append(temp)
                    found_temperature = temp
                    self._last_temperature_feedback = temp
                    nearest_dist = self.db.get_nearest_distance(wx, wz)

                    # 间距阈值去重交由 database 负责（需求第十四节）
                    # 温度关闭时以 0.0 占位存储，不伪造触觉反馈
                    stored_temp = temp if temp is not None else 0.0
                    if self.db.add_point(wx, wz, stored_temp):
                        is_new_valid_point = True
                        self.deep_sea_points.append((wx, wz, stored_temp))

                        if stage == 3:
                            # 需求第一节第7条：方向与步长记忆
                            self.agent.record_discovery(action_taken)
                            # 远跳机制：新点确认后重置无进展计数
                            self._stage3_idle_steps = 0

                        # 需求第一节第8条：自适应搜索粒度 + 自主温度分析的数据注入
                        self._sync_finder_density()
                        self._sync_strategy_reference_temps()

                        # 需求第十节：详细版才输出实时查询过程
                        if self.is_detailed_mode:
                            temp_str = f"{temp:.4f}" if temp is not None else "无"
                            print(f"  [实时查询] 深海点命中 ({wx}, {wz}) 温度={temp_str} "
                                  f"累计={len(self.deep_sea_points)} "
                                  f"查询={self.cubiomes.query_count}/{max_queries}")
                    # 需求第二节：一旦发现任一深海点，立即终止当前区块扫描
                    break

            # === 记录全区块温度样本（温度, 该区块是否含深海）===
            # 这是"温度 → 深海概率"统计的数据基础：非深海区块提供分母。
            if self.config["temperature_enabled"] and chunk_temperature is not None:
                self.temperature_samples.append((chunk_temperature, chunk_has_deep))

            # === 状态转移 ===
            # 需求第十五节：发现深海点后，下一状态的温度使用实际采样点温度而非区块中心温度；
            # 未发现深海点时，使用本区块中心温度作为触觉反馈（否则温度感知无法参与状态编码）
            state_temperature = found_temperature if found_temperature is not None else chunk_temperature
            next_state = self.state_manager.get_state(
                *self.current_chunk, temperature=state_temperature
            )

            # 好奇心时序约束（state_manager.py 文档）：必须先 is_new_state 再 mark_visited
            is_new_state = self.state_manager.is_new_state(next_state)

            # === 奖励计算与Q-learning更新（仅Stage3）===
            if stage == 3:
                reward = self._calculate_reward(
                    is_new_deep_sea=is_new_valid_point,
                    distance_to_nearest=nearest_dist,
                    is_new_state=is_new_state,
                    is_out_of_bounds=False,
                    is_repeated_visit=not is_new_state,
                )
                self._learn(action_taken, reward, next_state, step_index=step_index)
                last_experience = (self.current_state, action_taken, reward, next_state, False)

            self.state_manager.mark_visited(next_state)
            self.current_state = next_state
            convergence_data.append({
                "query_count": self.cubiomes.query_count,
                "deep_sea_points": len(self.deep_sea_points),
                "stage": stage,
            })

            # 记录阶段切换点，供收敛分析使用
            new_stage = self._get_current_stage()
            if new_stage != stage:
                self._stage_transitions.append({
                    "from_stage": stage,
                    "to_stage": new_stage,
                    "query_count": self.cubiomes.query_count,
                    "deep_sea_points": len(self.deep_sea_points),
                })

        # episode 终止：对最后一步补 done=True 更新
        if stage == 3 and last_experience is not None:
            s, a, r, ns, _ = last_experience
            self.agent.update(s, a, r, ns, True)

        result = {
            "seed": seed,
            "total_queries": self.cubiomes.query_count,
            "deep_sea_points_found": len(self.deep_sea_points),
            "deep_sea_points": self.deep_sea_points.copy(),
            "final_stage": stage,
            "epsilon": self.agent.epsilon,
            "convergence_data": convergence_data,
            "temperature_history": self.temperature_history.copy(),
            "temperature_histogram": self._build_temperature_histogram(),
            # 需求第一节第6条：温度 → 深海概率，AI 自主识别深海倾向温度区间
            "temperature_deep_probability": self.build_temperature_deep_probability(),
            "temperature_preference": self._analyze_temperature_preference(),
            "temperature_samples": list(self.temperature_samples),
            "stage_transitions": list(self._stage_transitions),
            "visited_states": self.state_manager.visited_count,
            "visited_chunks": len(self._visited_chunks),
            # 温度探针次数：temp_on 与 temp_off 两组此值应严格相等，
            # 供下游脚本核对消融预算公平性（两组扫描区块数一致）
            "temperature_probes": self._temperature_probe_count,
        }

        # 需求第十节：输出双版本 —— 详细版输出本次完整结果，简易版仅输出一行概要
        if self.is_detailed_mode:
            print(f"[详细版] Episode 完整结果 | 种子={seed} | 总查询={self.cubiomes.query_count} | "
                  f"深海点={len(self.deep_sea_points)} | 最终阶段=Stage{stage}")
            print(f"         epsilon={self.agent.epsilon:.4f} | "
                  f"已访问状态={self.state_manager.visited_count} | "
                  f"已扫描区块={len(self._visited_chunks)}")
            print(f"         深海点坐标: {[f'({p[0]},{p[1]})' for p in self.deep_sea_points]}")
            if self.config["temperature_enabled"]:
                pref = result["temperature_preference"]
                if pref:
                    print(f"         自主温度分析: 区块样本={pref['sample_count']} "
                          f"(深海{pref['deep_sample_count']}/非深海{pref['non_deep_sample_count']}) "
                          f"温度均值={pref['mean_temperature']:.4f} "
                          f"区间=[{pref['min_temperature']:.4f}, {pref['max_temperature']:.4f}]")
                    print(f"         AI 发现的深海倾向温度区间: {pref['best_temperature_bin']} "
                          f"(概率 {pref['best_bin_probability']:.2%}, "
                          f"{pref['best_bin_deep']}/{pref['best_bin_total']} 个区块)")
                    if "deep_mean_temperature" in pref:
                        print(f"         深海区块温度均值={pref['deep_mean_temperature']:.4f} vs "
                              f"非深海区块温度均值={pref['non_deep_mean_temperature']:.4f} "
                              f"(温差 {pref['temperature_gap']:.4f})")
                # 需求第一节第6条：绘制「温度 → 深海概率」直方图
                self.print_temperature_deep_probability()
            if self._stage_transitions:
                for tr in self._stage_transitions:
                    print(f"         阶段切换: Stage{tr['from_stage']} -> Stage{tr['to_stage']} "
                          f"@查询{tr['query_count']} (深海点{tr['deep_sea_points']})")
        else:
            print(f"[简易版] 种子={seed} | 深海点={len(self.deep_sea_points)} | "
                  f"查询={self.cubiomes.query_count} | Stage{stage}")

        return result

    # ==================== 批量实验入口（需求第十四节） ====================

    def run_experiment(self, world_seeds: List[int], repeats: int,
                       algo_seed_start: int = 42) -> dict:
        """
        批量运行 world_seeds × repeats 次 episode 并聚合统计。
        需求第十二节：固定地形种子控制环境方差，重复运行评估算法方差。

        输出双版本由 config["output_mode"] 决定（需求第十节）：
            simple   -> 简易版：每个世界种子输出一行平均值（20个种子即20行）
            detailed -> 详细版：输出每次运行的完整结果 + 实时查询输出

        ⚠️ 需求第十五节：CSV 生成与图表绘制绝不在此文件中，本方法只返回统计字典。

        Args:
            world_seeds: 世界种子列表（控制 cubiomes 地形生成）
            repeats: 每个世界种子的算法重复次数
            algo_seed_start: 算法随机种子起始值（控制 epsilon-greedy 等内部随机性）
        Returns:
            含 mean/std/n_runs/success_rate 与全部原始结果的统计字典

            success_rate 口径 = 平均深海点数 / 目标深海点数（完成度），
            而非「跑满目标的次数占比」。完成度口径在消融实验中能保留
            各配置之间的细微差距；若用达标次数口径，多数配置会同时显示 0%，
            丢失全部区分度。target_reached_count 另行保留达标次数供参考。
        """
        all_points: List[int] = []
        all_queries: List[int] = []
        raw_results: List[dict] = []
        algo_seed_counter = algo_seed_start
        detailed = self.is_detailed_mode
        target_points = self.config["target_deep_sea_points"]

        if detailed:
            print(f"[详细版] 开始批量实验：{len(world_seeds)} 个世界种子 × {repeats} 次重复 "
                  f"= {len(world_seeds) * repeats} 次运行")

        for world_seed in world_seeds:
            seed_points: List[int] = []
            seed_queries: List[int] = []
            seed_target_reached = 0

            for repeat_index in range(repeats):
                # 分离世界种子与算法种子（需求第十二节）
                if hasattr(self.cubiomes, "set_seed"):
                    self.cubiomes.set_seed(world_seed)

                algo_seed = algo_seed_counter
                algo_seed_counter += 1

                try:
                    result = self.run_single_episode(seed=algo_seed)
                    points = result["deep_sea_points_found"]
                    queries = result["total_queries"]
                except Exception as e:
                    print(f"\n  [异常] world_seed={world_seed} algo_seed={algo_seed} 失败: {e}",
                          file=sys.stderr)
                    result = {"world_seed": world_seed, "algo_seed": algo_seed,
                              "deep_sea_points_found": 0, "total_queries": 0, "error": str(e)}
                    points, queries = 0, 0

                seed_points.append(points)
                seed_queries.append(queries)
                all_points.append(points)
                all_queries.append(queries)

                if points >= target_points:
                    seed_target_reached += 1

                result["world_seed"] = world_seed
                result["algo_seed"] = algo_seed
                raw_results.append(result)

                # 详细版：输出每次运行的完整结果（实时查询输出已在 run_single_episode 内打印）
                if detailed:
                    print(f"  [详细版] 世界种子={world_seed} 第{repeat_index + 1}/{repeats}次 | "
                          f"算法种子={algo_seed} | 深海点={points} | 查询={queries} | "
                          f"Stage{result.get('final_stage')} | "
                          f"epsilon={result.get('epsilon', float('nan')):.4f}")

            # 简易版：每个世界种子输出一行平均值（20个种子 => 20行）
            if not detailed and seed_points:
                avg_p = sum(seed_points) / len(seed_points)
                avg_q = sum(seed_queries) / len(seed_queries)
                # 成功率 = 平均深海点数 / 目标深海点数
                seed_rate = avg_p / target_points
                print(f"  [简易版] 世界种子 {world_seed:>2} | 平均深海点={avg_p:>6.2f} | "
                      f"平均查询={avg_q:>7.1f} | "
                      f"成功率={avg_p:.2f}/{target_points} ({seed_rate * 100:>5.1f}%)")

        n = len(all_points)
        mean_points = sum(all_points) / n if n else 0.0
        mean_queries = sum(all_queries) / n if n else 0.0
        std_points = math.sqrt(sum((p - mean_points) ** 2 for p in all_points) / n) if n else 0.0
        std_queries = math.sqrt(sum((q - mean_queries) ** 2 for q in all_queries) / n) if n else 0.0

        # 成功率 = 平均深海点数 / 目标深海点数（完成度口径，便于消融实验横向对比）
        success_rate = (mean_points / target_points) if target_points > 0 else 0.0
        # 辅助指标：真正把目标跑满的次数（原 success_count 口径，保留供参考）
        target_reached_count = sum(1 for p in all_points if p >= target_points)

        total_stats = {
            "mean_points": mean_points,
            "std_points": std_points,
            "mean_queries": mean_queries,
            "std_queries": std_queries,
            "n_runs": n,
            "success_rate": success_rate,
            "target_reached_count": target_reached_count,
            "target_deep_sea_points": target_points,
            "raw_results": raw_results,
        }

        print(f"  [总体] 平均深海点={mean_points:.2f}±{std_points:.2f} | "
              f"平均查询={mean_queries:.1f}±{std_queries:.1f} | "
              f"成功率={mean_points:.2f}/{target_points} ({success_rate * 100:.1f}%) | "
              f"跑满目标次数={target_reached_count}/{n}")

        return total_stats
