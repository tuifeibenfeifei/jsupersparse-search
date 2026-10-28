
"""
main_algorithm.py
Core RL backbone: orchestrates macro chunk traversal, micro coordinate selection, state management, reward computation, and Q-learning updates

[Role] Req. Sec. 15 Stage 3: this file is a "dumb" pure executor.
    - Purely config-driven: all strategies are instantiated dynamically via a factory from YAML strings
    - No loops / no exports: it only provides run_single_episode() returning a result dict,
      20 repeats, CSV generation and plotting never happen in this file

[Fix rationale] (<=100 chars)
1. Reward call aligned with the real FullRewardBaseline.calculate signature: the old triple try/fallback passed
   rejected arguments such as current_state/visited_states/current_chunk, all three raised TypeError,
   so the reward was always 0.0 and Stage 3 received no learning signal.
2. Temperature injection into micro point selection get_points(temperature=): the old call omitted it, permanently degrading adaptive_sampling.
3. On episode reset, finder/strategy are rebuilt and reset(): the old internal cursors accumulated across episodes,
   so when the same Explorer ran 400 times, scans after the 1st no longer started from the origin.
4. Added deduplication of visited chunks: macro traversal regenerated the same chunk and the old code re-queried it, wasting budget.
5. Curiosity now uses the StateManager.is_new_state/mark_visited contract; the duplicated visited_states was removed.
6. Temperature histogram bin ranges are now computed from observed data instead of being hard-coded, per Requirement Sec. 1 (no prior knowledge).
"""

import random
import math
import sys
from typing import Tuple, Dict, List, Optional
from collections import deque, Counter

# ==================== Defensive imports (fix ImportError) ====================
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


# ==================== Default configuration fallback ====================
# Req. Sec. 15: keep this dict as defaults when the YAML is missing keys.
# Keys strictly match the YAML files under configs/ (Req. Sec. 10: config-driven; no hard-coded ablation values).
STAGE1_BASELINE = {
    # --- Experiment id (Req. Sec. 10: the dynamic header must print the algorithm name) ---
    "experiment_name": "Deep-sea biome exploration - black-box learning framework",
    "output_mode": "simple",             # simple = summary mode (per-seed averages) / detailed = full 20-run results + live query output

    # --- Perception layer (Stage 1 ablation variable) ---
    # temperature_enabled: master switch for temperature. False = pure blind search (no temperature query / bias / state).
    "temperature_enabled": True,
    # temperature_in_state: whether temperature enters the Q-state encoding (independent sub-switch; only active when temperature_enabled=True).
    #   Measured: temperature has no localization power for "is deep sea" (AUC=0.458 in ocean), and putting it in the Q-state inflates the state space,
    #   diluting learning (old temp_on 31.23 < temp_off 32.43); yet temperature predicts the ocean biome exactly (Spearman=1.0),
    #   and the deep-sea rate across temperature bands is non-monotonic (frozen 48% > normal 24% > cold 18% > lukewarm 0%),
    #   so macro guidance bias is effective (+0.79). Final architecture: temperature does NOT enter the Q-state (this key=False); it is used only for guidance bias.
    "temperature_in_state": False,
    "temperature_bins": 10,              # bin count for the temperature histogram only; range comes from observed data
    "temperature_precision": 2,          # temperature quantization precision for state encoding (decimal places), not the temperature range
    # State-grid spatial granularity (in chunks). The temperature layer has a resolution of 16 chunks; this must be >16 to add information;
    # if <=16, temperature is derivable from chunk coordinates, so the temperature on/off ablation yields identical results.
    "spatial_granularity": 32,
    # Default micro point strategy center_point: best measured in Stage 1 (32.84) with the lowest query cost (9138).
    "micro_strategy": "center_point",

    # --- Temperature-guided search (Req. Sec. 1 #10 + Sec. 4 #27: Stage 3 drives search from learned temperature regularities) ---
    # Orthogonal to temperature_enabled (state encoding): this switch lets the agent use "self-learned temperature band -> deep-sea probability"
    # to bias Stage 3 macro actions; temperature never enters the reward (Req. #6/30) and this only takes effect in Stage 3 (Req. #26).
    # Measured basis: frozen band deep-sea rate 48.4% (1.77x baseline), lukewarm/warm bands ~0%, and the relation is non-monotonic
    # (cold 17.9% < normal 23.5%), so a hand-crafted "colder is better" heuristic fails; the probability table must be learned.
    "temperature_guidance_enabled": True,
    "temperature_guidance_weight": 3.0,  # exploration prior weight (hyperparameter, sweepable in Stage 4)
    "temperature_guidance_min_samples": 20,  # min samples before the probability table applies; below this, fall back to pure RL

    # --- Movement layer (Stage 2 ablation variable) ---
    "chunk_finder": "ray_scan",
    "deep_sea_distance_threshold": 416,

    # --- Reward layer (Stage 3 ablation variable) ---
    "deep_sea_reward_enabled": True,
    "curiosity_reward_enabled": True,
    "penalty_reward_enabled": True,
    "deep_sea_reward_value": 10.0,
    "curiosity_reward_value": 3.0,
    "curiosity_weight": 1.0,
    "out_of_bounds_penalty": -10.0,
    "repeat_visit_penalty": -5.0,

    # --- RL hyperparameters ---
    "learning_rate": 0.1,
    "discount_factor": 0.95,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay": 0.995,
    "replay_buffer_size": 8192,          # Req. Sec. 1: fixed at 8192
    "replay_batch_size": 16,
    "replay_frequency": 5,               # trigger one replay every N steps

    # --- Stage-switch thresholds (Req. Sec. 4) ---
    "stage2_min_deep_sea_points": 5,
    "stage3_min_deep_sea_points": 10,

    # --- Global limits (Req. Sec. 1 #9: exact values determined by the Stage 4 parameter sweep) ---
    "max_queries_per_episode": 10000,
    "target_deep_sea_points": 40,
    "search_radius": 10000,

    # --- Action space (Req. Sec. 9: 32 actions = 8 directions x 4 step sizes) ---
    "action_directions": 8,
    "action_steps": [1, 2, 4, 8],
    "direction_memory_bonus": 0.5,       # Req. Sec. 1: weight for direction & step memory

    # --- Stage 3 long-jump mechanism (completing _get_start_from_deep_point from the zhongxin reference) ---
    # Rationale: Stage 3 action steps are only 16-128 chunks, far smaller than the deep-sea exclusion radius (deep_sea_distance_threshold);
    # local hits all land inside the same exclusion radius and are deduplicated by the database, so the budget dies near already-found points
    # (measured: Stage 3 consumed 95%+ of the budget with a 0.74% useful-hit rate). The jump leaves a found deep-sea point in a random direction,
    # moving threshold x [min_ratio, max_ratio] chunks to escape the exclusion radius and reach new regions;
    # all jump distances derive from deep_sea_distance_threshold (auto-adapts when sweeping the threshold; zero hard-coding).
    "stage3_jump_enabled": True,         # long-jump switch
    "stage3_jump_interval": 30,          # trigger a long jump after N consecutive Stage 3 steps without a new deep-sea point
    "stage3_jump_min_ratio": 1.25,       # min jump = spacing threshold x this ratio (>1 guarantees leaving the exclusion radius)
    "stage3_jump_max_ratio": 5.0,        # max jump = spacing threshold x this ratio
}


def _merge_config(user_config: dict, default_config: dict) -> dict:
    """Recursively merge user config over default config; user values win."""
    merged = default_config.copy()
    for key, value in user_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_config(value, merged[key])
        else:
            merged[key] = value
    return merged


class ReplayBuffer:
    """Fixed-capacity FIFO experience replay buffer (Req. Sec. 1: capacity fixed at 8192)"""

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
    """Q-learning agent"""

    def __init__(self, config: dict):
        self.lr = config["learning_rate"]
        self.gamma = config["discount_factor"]
        self.epsilon_start = config["epsilon_start"]
        self.epsilon = self.epsilon_start
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_decay = config["epsilon_decay"]
        self.q_table: Dict[Tuple, List[float]] = {}

        # action space derived from config; hard-coding forbidden (Req. Sec. 10)
        self.num_directions = int(config["action_directions"])
        self.action_steps: List[int] = list(config["action_steps"])
        self.num_actions = self.num_directions * len(self.action_steps)

        # Req. Sec. 1: record (direction, step) pairs that led to deep-sea finds and weight them higher in later decisions
        self.direction_step_memory: Dict[Tuple[int, int], int] = {}
        self.memory_bonus = float(config.get("direction_memory_bonus", 0.5))

    def reset(self):
        """Req. Sec. 1 #3: stateless reset; clear the Q-table and visit records"""
        self.q_table.clear()
        self.direction_step_memory.clear()
        self.epsilon = self.epsilon_start

    def decode_indices(self, action: int) -> Tuple[int, int]:
        """action index -> (direction index, step index)"""
        return action % self.num_directions, action // self.num_directions

    def record_discovery(self, action: int):
        key = self.decode_indices(action)
        self.direction_step_memory[key] = self.direction_step_memory.get(key, 0) + 1

    def get_q_values(self, state: Tuple) -> List[float]:
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.num_actions
        return self.q_table[state]

    def select_action(self, state: Tuple, is_stage3: bool = False) -> int:
        """epsilon-greedy action selection (with direction-memory bonus). Behavior identical to before refactoring."""
        return self.select_action_with_bias(state, None, apply_memory=is_stage3)

    def select_action_with_bias(self, state: Tuple, action_bias, apply_memory: bool = True) -> int:
        """
        epsilon-greedy plus an external action bias (e.g. the temperature-guided exploration prior).

        action_bias: callable(action_idx) -> float, added on top of the Q value and the direction-memory bonus.
        The bias **does not change the epsilon exploration probability**; it only affects the ranking of actions during "exploitation" -
        it is therefore an exploration prior, not a reward; temperature never enters any reward formula (Req. #6/30).
        Ties are broken at random, preserving the randomness of the original select_action.
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
    """Main deep-sea exploration class"""

    # 8-direction displacement table, order matches RayScanFinder._DIRECTIONS:
    # E, SE, S, SW, W, NW, N, NE
    _DIRECTION_DELTAS: List[Tuple[int, int]] = [
        (1, 0), (1, -1), (0, -1), (-1, -1),
        (-1, 0), (-1, 1), (0, 1), (1, 1),
    ]

    def __init__(self, config: dict):
        self.config = _merge_config(config or {}, STAGE1_BASELINE)
        self.cubiomes = CubiomesWrapper()

        # internal state fields must be initialized before the factory methods:
        # _create_micro_strategy() reads self._algo_seed so randomized strategies are reproducible
        self._algo_seed: Optional[int] = None
        self._last_temperature_feedback: Optional[float] = None
        self._visited_chunks: set = set()
        self._stage_transitions: List[Dict[str, int]] = []
        # temperature probe count: equal between temp_on and temp_off (evidence of budget fairness)
        self._temperature_probe_count: int = 0
        # temperature-grid cache {(gx,gz): temp}: caches temperatures while scanning chunks so Stage 3 bias costs zero extra queries
        self._temperature_grid_cache: Dict[Tuple[int, int], float] = {}
        # Stage 3 long-jump: consecutive steps without a new deep-sea point (triggers jump at stage3_jump_interval)
        self._stage3_idle_steps: int = 0

        # initialize the database with the configured threshold (Req. Sec. 3)
        self.db = DeepSeaDatabase(self.config["deep_sea_distance_threshold"])

        # temperature enters the Q-state encoding only when both the master and sub switches are true.
        # with temperature_in_state=False: temperature is still queried and still biases guidance, but does not enter the state.
        self._temp_in_state = bool(
            self.config.get("temperature_enabled", True)
            and self.config.get("temperature_in_state", False)
        )
        self.state_manager = StateManager(
            temperature_enabled=self._temp_in_state,
            temperature_precision=self.config.get("temperature_precision", 2),
            spatial_granularity=self.config.get("spatial_granularity", 32),
        )
        # config sanity check: only when temperature enters the state must granularity exceed the temperature resolution (16 chunks);
        # otherwise temperature is derivable from chunk coordinates and contributes nothing to the state space.
        if self._temp_in_state and not self.state_manager.granularity_is_coarse_enough:
            print(f"[WARNING] spatial_granularity={self.state_manager.spatial_granularity} "
                  f"is not greater than the temperature resolution {self.state_manager.temperature_resolution_chunks} chunks,"
                  f"temperature is derivable from chunk coordinates, contributing nothing to the state space",
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
        # full-grid temperature samples [(temp, whether this chunk contains deep sea), ...]
        # Req. Sec. 1 #6: only with both deep-sea and non-deep-sea samples can the "temperature -> deep-sea probability" be computed,
        # only then can the agent discover which temperature ranges favor deep sea. Recording deep-sea temperatures alone cannot form a probability.
        self.temperature_samples: List[Tuple[float, bool]] = []

        self._print_header()

    # ==================== Factory methods (Req. Sec. 15: dynamic string-based instantiation) ====================

    def _create_finder(self) -> BaseFinder:
        finder_map = {
            "ray_scan": RayScanFinder,
            "spiral_scan": SpiralScanFinder,
            "adaptive_scan": AdaptiveScanFinder,
            "grid_scan": GridScanFinder,
            "anchor_scan": AnchorScanFinder,
        }
        name = self.config["chunk_finder"]
        # unknown strategy names must raise explicitly: silently falling back to ray_scan would let ablations "complete" under wrong configs,
        # producing plausible-looking but mismatched results with no warning (bottom line of config-driven design).
        if name not in finder_map:
            raise KeyError(
                f"unknown macro traversal strategy '{name}', available: {sorted(finder_map.keys())}"
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
        # inject the algorithm seed so randomized strategies are reproducible (Req. Sec. 12: separate environment and algorithm randomness)
        kwargs = dict(self.config)
        kwargs["seed"] = self._algo_seed
        return strategy_class(**kwargs)

    # ==================== Output header (Req. Sec. 10: dynamic description header) ====================

    @property
    def is_detailed_mode(self) -> bool:
        """Req. Sec. 10: detailed mode prints 20 full runs + live query output; simple mode prints only averages"""
        return str(self.config.get("output_mode", "simple")).lower() == "detailed"

    def _reward_switch_states(self) -> str:
        """
        4 reward switch states (Req. Sec. 5: reward modules split into 4 independent files)
        deep-sea / curiosity / penalty sub-switches plus the all-on baseline aggregator state
        """
        rm = self.reward_module
        deep = "ON" if rm.deep_sea.enabled else "OFF"
        curi = "ON" if rm.curiosity.enabled else "OFF"
        pen = "ON" if rm.penalty.enabled else "OFF"
        # all-on baseline: only when all three sub-rewards are enabled is it the "all-on control group"
        all_on = "ALL-ON control" if (rm.deep_sea.enabled and rm.curiosity.enabled and rm.penalty.enabled) else "PARTIAL-OFF"
        return (f"deep-sea={deep} | curiosity={curi} | penalty={pen} | baseline={all_on}")

    def _print_header(self):
        """
        Req. Sec. 10: on startup the program must read the current YAML and dynamically print a header,
        covering algorithm name, spacing, sampling, traversal, the 4 reward switches, temperature switches, and the query cap.
        """
        cfg = self.config
        rm = self.reward_module
        temp_status = "ON (black-box sensing)" if cfg["temperature_enabled"] else "OFF (blind search)"

        print("\n" + "=" * 74)
        print(f"  algorithm       : {cfg.get('experiment_name')}")
        print(f"  output mode    : {'DETAILED (20 full runs + live query output)' if self.is_detailed_mode else 'SIMPLE (per-seed averages)'}")
        print("-" * 74)
        print(f"  macro traversal: {cfg['chunk_finder']}")
        print(f"  micro sampling : {cfg['micro_strategy']}")
        print(f"  temperature    : {temp_status}")
        # temperature-guided search (Req. Sec. 4 #27): only active in Stage 3, requires temperature sensing on
        if not cfg["temperature_enabled"]:
            guidance_status = "OFF (temperature sensing off, bias auto-disabled)"
        elif cfg.get("temperature_guidance_enabled", False):
            guidance_status = (f"ON (Stage 3 probability bias, weight {cfg.get('temperature_guidance_weight')}, "
                               f"min samples {cfg.get('temperature_guidance_min_samples')})")
        else:
            guidance_status = "OFF"
        print(f"  temp guidance  : {guidance_status}")
        # whether temperature enters the Q-state encoding (orthogonal to guidance bias). Measured: entering the state hurts; biasing helps.
        if not cfg["temperature_enabled"]:
            state_temp_status = "OFF (temperature sensing off)"
        elif self._temp_in_state:
            state_temp_status = "ON (temperature quantized into the state encoding)"
        else:
            state_temp_status = "OFF (temperature NOT in Q-state; guidance bias only - final architecture)"
        print(f"  temp in state  : {state_temp_status}")
        print(f"  deep-sea spacing: {cfg['deep_sea_distance_threshold']} chunks")
        # state-encoding config: only when temperature enters the state must granularity exceed the temperature resolution (16 chunks)
        gran = self.state_manager.spatial_granularity
        prec = self.state_manager.temperature_precision
        if self._temp_in_state:
            gran_note = ("valid: coarser than the 16-chunk temperature resolution"
                         if self.state_manager.granularity_is_coarse_enough
                         else "WARNING: not coarser than the 16-chunk temperature resolution")
        else:
            gran_note = "temperature not in state; granularity only partitions coordinates"
        print(f"  state granularity: {gran} chunks/cell ({gran_note})")
        print(f"  temp precision : {prec} decimal places (resolution parameter, not the temperature range)")
        print("-" * 74)
        print(f"  4 reward switches:")
        print(f"    {self._reward_switch_states()}")
        print(f"    deep-sea reward  : +{rm.deep_sea.reward}")
        print(f"    curiosity reward: +{rm.curiosity.reward} (weight {rm.curiosity.weight})")
        print(f"    out-of-bounds   : {rm.penalty.out_of_bounds}")
        print(f"    repeated-visit  : {rm.penalty.repeated_visit}")
        print("-" * 74)
        print(f"  query cap     : {cfg['max_queries_per_episode']} / target {cfg['target_deep_sea_points']} deep-sea points")
        print(f"  search radius : {cfg['search_radius']} chunks (square region, X/Z only)")
        print(f"  replay buffer : {cfg['replay_buffer_size']}")
        print(f"  stage gates   : Stage 2 >= {cfg['stage2_min_deep_sea_points']} pts, "
              f"Stage 3 >= {cfg['stage3_min_deep_sea_points']} pts")
        print(f"  action space  : {cfg['action_directions']} dirs x {len(cfg['action_steps'])} steps "
              f"= {self.agent.num_actions} actions, steps={cfg['action_steps']}")
        print("=" * 74 + "\n")

    # ==================== State reset (Req. Sec. 1: stateless reset) ====================

    def _reset_state(self):
        """Clear all state before each experiment so no history leaks across runs"""
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

        # zero the black-box query counter; it is the single accounting basis for the query budget
        self.cubiomes.query_count = 0

        # [KEY] rebuild the finder and micro_strategy and reset their internal cursors.
        # if RayScanFinder._step / SpiralScanFinder._x,_z etc. are not cleared,
        # the same Explorer would keep expanding from the previous end point across episodes, contaminating them.
        self.finder = self._create_finder()
        self.micro_strategy = self._create_micro_strategy()
        self.finder.reset()

        self.current_chunk = (0, 0)
        self.current_state = self.state_manager.get_state(0, 0, temperature=None)
        self.state_manager.mark_visited(self.current_state)

    # ==================== Stages and actions ====================

    def _get_current_stage(self) -> int:
        """Req. Sec. 4: three-stage learning mechanism"""
        num_points = len(self.deep_sea_points)
        if num_points >= self.config["stage3_min_deep_sea_points"]:
            return 3
        elif num_points >= self.config["stage2_min_deep_sea_points"]:
            return 2
        return 1

    def _decode_action(self, action: int) -> Tuple[int, int, int, int, int]:
        """action index -> (direction index, step index, step value, dx, dz)"""
        dir_idx, step_idx = self.agent.decode_indices(action)
        step = self.config["action_steps"][step_idx]
        dx, dz = self._DIRECTION_DELTAS[dir_idx]
        return dir_idx, step_idx, step, dx, dz

    def _is_out_of_bounds(self, x: int, z: int) -> bool:
        """Req. Sec. 2: confined to a square of radius search_radius chunks centered on the spawn point"""
        radius = self.config["search_radius"]
        return abs(x) > radius or abs(z) > radius

    # ==================== Feedback injection ====================

    def _sync_finder_density(self):
        """Req. Sec. 1 #8: inject deep-sea density into the macro strategy for adaptive search granularity"""
        updater = getattr(self.finder, "update_deep_sea_density", None)
        if updater is None:
            return
        chunks = [(x // 16, z // 16) for x, z, _ in self.deep_sea_points]
        updater(chunks)

    def _sync_strategy_reference_temps(self):
        """Req. Sec. 9: adaptive_sampling must receive temperature feedback as an argument"""
        setter = getattr(self.micro_strategy, "set_reference_temps", None)
        if setter is None:
            return
        temps = [t for _, _, t in self.deep_sea_points if t is not None]
        setter(temps)

    # ==================== Temperature probe (guarantees fair ablation budgets) ====================

    def _probe_temperature(self, x: int, z: int) -> Optional[float]:
        """
        Unified temperature query entry. **The query always runs and always counts against the budget, regardless of temperature_enabled**,
        only discarding the result when the switch is off (returns None, i.e. temperature is "absent").

        Why this design is required:
            if queries were skipped when temperature is off, temp_off would consume 1 fewer query per chunk than temp_on,
            letting it scan about 22% more chunks under the same budget (measured with corners_and_center).
            the group difference would then mix "the effect of temperature features" with "the difference in scanned chunks",
            making attribution impossible; the core Stage 1 ablation would lose its control-variable meaning.
            making both groups pay identical query costs leaves "whether temperature enters the state encoding" as the only difference.

        Returns:
            temperature_enabled=True  -> actual temperature float (None on query failure)
            temperature_enabled=False -> always None (query ran and was billed; the result is discarded)
        """
        if self.cubiomes.query_count >= self.config["max_queries_per_episode"]:
            return None

        try:
            temp = self.cubiomes.get_temperature(x, z)
        except Exception as e:
            print(f"[WARNING] get_temperature({x},{z}) failed: {e}", file=sys.stderr)
            return None

        self._temperature_probe_count += 1
        # temperature ablation off: result does not enter state, statistics, or strategy; temperature is "absent"
        if not self.config["temperature_enabled"]:
            return None
        return temp

    # ==================== Temperature statistics (Req. Sec. 1 #6: autonomous temperature analysis) ====================

    def _valid_temperatures(self) -> List[float]:
        """deep-sea point temperature samples (missing/NaN filtered out)"""
        return [t for t in self.temperature_history
                if t is not None and not (isinstance(t, float) and math.isnan(t))]

    def _valid_temperature_samples(self) -> List[Tuple[float, bool]]:
        """full-grid temperature samples [(temp, is_deep_sea)], missing/NaN filtered out"""
        return [(t, d) for t, d in self.temperature_samples
                if t is not None and not (isinstance(t, float) and math.isnan(t))]

    def _build_temperature_bins(self) -> List[Tuple[str, float, float]]:
        """
        Split into equal-width bins over the temperature range (min~max) actually observed in this episode.
        No temperature interval is hard-coded - consistent with Req. Sec. 1 "no prior knowledge" and
        item 6 "autonomous identification... no manual thresholds".
        Returns: [(bin label, lower bound, upper bound), ...]
        """
        samples = self._valid_temperature_samples()
        if not samples:
            return []

        bins = max(int(self.config.get("temperature_bins", 10)), 1)
        lo = min(t for t, _ in samples)
        hi = max(t for t, _ in samples)
        if hi <= lo:
            # all samples share the same temperature; use a single bin
            return [(f"{lo:.2f}~{lo:.2f}", lo, hi)]

        width = (hi - lo) / bins
        edges = []
        for i in range(bins):
            b_lo = lo + i * width
            b_hi = lo + (i + 1) * width
            edges.append((f"{b_lo:.2f}~{b_hi:.2f}", b_lo, b_hi))
        return edges

    def _bin_index(self, temp: float, edges: List[Tuple[str, float, float]]) -> int:
        """map a temperature value to a bin index (last bin closed on the right, includes the upper bound)"""
        n = len(edges)
        for i in range(n - 1):
            if temp < edges[i][2]:
                return i
        return n - 1

    def _build_temperature_histogram(self) -> Dict[str, int]:
        """
        Temperature histogram: total samples per bin (deep sea and non-deep sea combined).
        Keeps original keys and semantics for backward compatibility of the result dict.
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
        Req. Sec. 1 #6: automatically compute the "temperature -> deep-sea probability" histogram data,
        autonomously identifying temperature ranges that favor deep sea without manual thresholds.

        probability = deep-sea chunks in this bin / all chunks in this bin
        (the denominator comes from "every chunk is temperature-queried", which is why full-grid temperature sampling is required)

        Returns:
            {bin label: {"deep": deep-sea count, "non_deep": non-deep-sea count, "total": total,
                      "probability": deep-sea probability}}
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
        Autonomously identify temperature ranges favoring deep sea (Req. Sec. 1 #6).
        Only the accumulated data decides the highest-probability bin; no manual prior thresholds.
        """
        prob_table = self.build_temperature_deep_probability()
        samples = self._valid_temperature_samples()
        if not prob_table or not samples:
            return {}

        temps = [t for t, _ in samples]
        deep_temps = [t for t, d in samples if d]
        non_deep_temps = [t for t, d in samples if not d]

        # the highest-probability bin = the temperature range the agent autonomously discovered
        # require a minimum sample size per bin to avoid small-sample bins reporting inflated probability by chance.
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

        # mean-temperature contrast between deep and non-deep chunks, used to judge whether temperature is discriminative
        if deep_temps and non_deep_temps:
            deep_mean = sum(deep_temps) / len(deep_temps)
            non_deep_mean = sum(non_deep_temps) / len(non_deep_temps)
            analysis["deep_mean_temperature"] = deep_mean
            analysis["non_deep_mean_temperature"] = non_deep_mean
            analysis["temperature_gap"] = abs(deep_mean - non_deep_mean)

        return analysis

    def print_temperature_deep_probability(self) -> None:
        """
        Print the "temperature -> deep-sea probability" histogram (Req. Sec. 1 #6: autonomous discovery, no manual thresholds)
        Only invoked in detailed output mode.
        """
        prob_table = self.build_temperature_deep_probability()
        if not prob_table:
            print("  insufficient temperature samples; cannot draw the temperature->deep-sea probability histogram.")
            return

        total_samples = sum(v["total"] for v in prob_table.values())
        total_deep = sum(v["deep"] for v in prob_table.values())

        print(f"\n  === temperature -> deep-sea probability histogram (autonomous analysis, bins from observed data) ===")
        print(f"  {'temp range':<16} {'deep':>7} {'non-deep':>9} {'prob':>9}  distribution")
        print(f"  {'─' * 62}")
        for label, info in prob_table.items():
            bar = '#' * int(info["probability"] * 30)
            print(f"  {label:<16} {info['deep']:>7} {info['non_deep']:>9} "
                  f"{info['probability']:>8.2%}  {bar}")
        print(f"  {'─' * 62}")
        if total_samples:
            print(f"  total samples: {total_samples} chunks | deep-sea chunks {total_deep} "
                  f"({total_deep / total_samples:.2%})")
        else:
            print("  total samples: 0")

        best_label = max(prob_table.items(), key=lambda kv: kv[1]["probability"])
        if best_label[1]["probability"] > 0:
            print(f"  [AI FOUND] the temperature range with the highest deep-sea probability is {best_label[0]},"
                  f"probability {best_label[1]['probability']:.2%}")

    # ============ Temperature-guided search (Req. Sec. 1 #10 + Sec. 4 #27) ============

    @staticmethod
    def _wilson_lower_bound(deep: int, total: int, z: float = 1.96) -> float:
        """
        Lower bound of the Wilson score 95% confidence interval.
        Estimates the lower bound of the deep-sea probability per temperature band: small-sample bands are automatically pulled down, avoiding inflated probability from chance hits,
        more robust than the point estimate deep/total (a 3-sample bin once inflated to 66.7%).

        Zero manual thresholds: z=1.96 is a statistical constant for the 95% confidence level, unrelated to "which temperature is closer to deep sea",
        and does not constitute the prior knowledge banned by Req. Sec. 1.
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
        Build the "temperature band -> Wilson lower bound of deep-sea probability" table and the base rate.
        Binning is fully data-driven (range from this episode's observed min~max, reusing _build_temperature_bins),
        with no manual thresholds such as frozen/cold/warm (Req. Sec. 1 "no prior knowledge").

        Returns:
            (band_lower_bounds: {bin index: Wilson lower bound}, base_rate: overall deep-sea rate)
            returns ({}, 0.0) when samples are insufficient.
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
        Temperature-grid coordinates: the temperature layer resolution is 16 chunks (256 blocks), constant within one cell.
        Source basis: cubiomes generator.h:57 (L_OCEAN_TEMP_256) and layers.c:3130 (x/8.0).
        Caches temperatures of queried chunks so action bias costs zero extra queries.
        """
        return chunk_x // 16, chunk_z // 16

    def _make_temperature_action_bias(self):
        """
        Build the Stage 3 temperature-guided action bias function (Req. Sec. 4 #27: drive search from learned temperature regularities).

        bias = weight x (Wilson lower bound of the target chunk's band - base rate)
          high-probability band (frozen measured at 1.77x) -> positive bias -> tend to move in and stay for fine search
          low-probability band (lukewarm/warm ~0) -> negative bias -> tend to leave
          unknown temperature cell/band -> 0 (neutral, falls back to pure RL)

        Three compliance red lines:
          1 zero manual thresholds: data-driven bands, Wilson lower bound, no manual temperature ranges.
          2 never in the reward: the bias only adds to the action-selection Q value (exploration prior); temperature stays out of rewards (Req. #6/30).
          3 only active in Stage 3: Stage 1 blind-search, Stage 2 collects but does not decide (Req. #25/26).

        Zero extra queries: the target chunk's temperature comes from _temperature_grid_cache (cached while scanning);
        uncached cells are treated as neutral; temperature is never queried extra for the bias.

        Returns:
            callable(action_idx)->float, or None (feature off / temperature off / insufficient samples).
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

    # ==================== Stage 3 long-jump mechanism (completing
    #  _get_start_from_deep_point from the zhongxin reference: jump outward from deep-sea points, escaping the 416-chunk exclusion radius) ====================

    def _stage3_far_jump(self) -> bool:
        """
        When Stage 3 makes no progress, jump a long distance from an already-found deep-sea point in a random direction.

        Why a long jump is needed (measured root cause):
            Stage 3 action steps are 16-128 chunks while the deep-sea exclusion radius = deep_sea_distance_threshold chunks.
            all local hits fall inside the exclusion radius of existing points and are deduplicated by the database; the useful-hit rate is only 0.74%,
            so the budget dies around already-found points. The long jump escapes the exclusion radius and reaches new regions.

        Jump distances derive entirely from the deep_sea_distance_threshold variable (auto-adapts when sweeping the threshold):
            min_dist = threshold x stage3_jump_min_ratio (>1, guarantees leaving the exclusion radius)
            max_dist = threshold × stage3_jump_max_ratio
        Same magnitude as zhongxin.py's [512, 1024] chunks (its exclusion radius is 512 chunks -> 1-2x the threshold).

        The target chunk must lie inside the search radius and be unvisited; after 16 failed retries, shrink toward the threshold and retry,
        still failing returns False (this round falls back to a normal action).

        Returns:
            True=jumped to a new target chunk; False=no jump
        """
        threshold = float(self.config["deep_sea_distance_threshold"])
        min_ratio = float(self.config.get("stage3_jump_min_ratio", 1.25))
        max_ratio = float(self.config.get("stage3_jump_max_ratio", 5.0))
        min_dist = max(int(threshold * min_ratio), int(threshold) + 32)  # chunks
        max_dist = int(threshold * max_ratio)
        if max_dist <= min_dist:
            max_dist = min_dist + 64

        # round 1: wide jump over threshold x [min_ratio, max_ratio]
        for _ in range(16):
            px, pz, _ = random.choice(self.deep_sea_points)
            dx, dz = random.choice(self._DIRECTION_DELTAS)
            dist = random.randint(min_dist, max_dist)
            nx = px // 16 + dx * (dist // 16)      # deep-sea point (chunks) -> chunk + chunk displacement
            nz = pz // 16 + dz * (dist // 16)
            if self._is_out_of_bounds(nx * 16, nz * 16):
                continue
            if (nx, nz) in self._visited_chunks:
                continue
            self.current_chunk = (nx, nz)
            return True

        # round 2: shrink toward the threshold to improve in-bounds hit rate
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

    # ==================== Learning updates ====================

    def _replay_batch_update(self):
        """Req. Sec. 1: experience replay - sample a batch from the buffer and replay it, breaking sample correlation"""
        batch_size = int(self.config.get("replay_batch_size", 16))
        if batch_size <= 0 or len(self.replay_buffer) < batch_size:
            return
        for s, a, r, ns, d in self.replay_buffer.sample(batch_size):
            self.agent.update(s, a, r, ns, d)

    def _learn(self, action: int, reward: float, next_state: Tuple,
               done: bool = False, step_index: int = 0):
        """single-step Q-learning update + experience push + frequency-based batch replay + epsilon decay"""
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
        Called with the real signature of FullRewardBaseline.calculate.
        Req. Sec. 5: total reward = deep-sea discovery + curiosity + penalty; temperature never participates.
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
            print(f"[WARNING] reward computation failed: {e}; reward set to 0 for this step", file=sys.stderr)
            return 0.0

    # ==================== Single episode ====================

    def run_single_episode(self, seed: int) -> dict:
        """
        Execute one complete search episode.
        Req. Sec. 15: this method only returns a result dict; 20 repeats / CSV / plots never happen in this file.
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

        # the query budget counts actual black-box calls (is_deep_ocean and get_temperature both bill)
        while self.cubiomes.query_count < max_queries and len(self.deep_sea_points) < target_points:
            stage = self._get_current_stage()
            action_taken = 0
            is_new_valid_point = False
            nearest_dist = 0.0
            found_temperature = None
            step_index += 1

            # === macro: fetch the next chunk ===
            if stage in (1, 2):
                # Req. Sec. 4: Stage 1 blind-search, Stage 2 only collects statistics without deciding; both scan with the preset step strategy
                cx, cz = self.current_chunk
                self.current_chunk = self.finder.get_next_chunk(cx, cz)
            else:
                # Req. Sec. 4: Stage 3 starts reward-driven efficient search
                # Req. Sec. 4 #27: Stage 3 drives efficient search from learned temperature regularities.
                # the temperature bias is an exploration prior added to the action-selection Q value; it never enters the reward (Req. #6/30).
                # _make_temperature_action_bias returns None when samples are insufficient or the feature is off, falling back to pure RL.
                self._stage3_idle_steps += 1
                # === long-jump: N consecutive steps without a new deep-sea point -> jump away from the exclusion radius ===
                # does not alter the action space / rewards / Q-learning; it is only an exploration restart (completing the reference implementation).
                if (self.config.get("stage3_jump_enabled", True)
                        and self._stage3_idle_steps >= self.config.get("stage3_jump_interval", 30)
                        and self.deep_sea_points
                        and self._stage3_far_jump()):
                    self._stage3_idle_steps = 0
                    # after a successful jump, current_chunk is updated; proceed directly to this round's query flow
                else:
                    temp_bias = self._make_temperature_action_bias()
                    action_taken = self.agent.select_action_with_bias(
                        self.current_state, temp_bias, apply_memory=True
                    )
                    _, _, step, dx, dz = self._decode_action(action_taken)
                    cx, cz = self.current_chunk
                    self.current_chunk = (cx + dx * step, cz + dz * step)

            chunk_x, chunk_z = self.current_chunk

            # === chunk dedup: skip already-scanned chunks to avoid burning the query budget ===
            # macro traversal regenerates identical chunk coordinates; without dedup the same chunks get queried repeatedly.
            if (chunk_x, chunk_z) in self._visited_chunks:
                next_state = self.state_manager.get_state(chunk_x, chunk_z, temperature=None)
                if stage == 3:
                    # Req. Sec. 5: repeated-visit penalty
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

            # === bounds check (Req. Sec. 2) ===
            if self._is_out_of_bounds(chunk_x * 16, chunk_z * 16):
                next_state = self.state_manager.get_state(0, 0, temperature=None)
                if stage == 3:
                    # penalties are computed by PenaltyReward; no hard-coded values here (Req. Sec. 10)
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

            # === full-grid temperature query (Req. Sec. 1 #6: the agent autonomously discovers the temperature-deep-sea relation) ===
            # every chunk's center is temperature-queried once; the (temperature, contains-deep-sea) pair is recorded regardless.
            # only with both deep-sea and non-deep-sea samples can "temperature -> deep-sea probability" be estimated,
            # letting the machine learn which ranges favor deep sea instead of only seeing deep-sea temperatures.
            # billed through _probe_temperature: when the switch is off the query still runs but the result is discarded,
            # guaranteeing temp_on/temp_off scan identical chunk counts (fair ablation budgets; see that method's doc).
            center_x, center_z = chunk_x * 16 + 8, chunk_z * 16 + 8
            chunk_temperature = self._probe_temperature(center_x, center_z)

            # write the temperature-grid cache: used by the Stage 3 bias with zero extra queries (None when off; not cached)
            if chunk_temperature is not None:
                self._temperature_grid_cache[
                    self._temperature_grid_key(chunk_x, chunk_z)
                ] = chunk_temperature

            # the current chunk's temperature is the feedback injected into the micro strategy (adaptive_sampling adjusts dynamically)
            # when temperature is off, chunk_temperature is always None; the micro strategy receives "no temperature" and falls back to geometric sampling
            self._last_temperature_feedback = chunk_temperature

            # === micro: point selection inside the chunk (with the chunk's temperature feedback) ===
            sample_points = self.micro_strategy.get_points(
                chunk_x, chunk_z, temperature=chunk_temperature
            )

            # === query each sampled point ===
            chunk_has_deep = False
            for wx, wz in sample_points:
                if self.cubiomes.query_count >= max_queries:
                    break

                try:
                    is_deep = self.cubiomes.is_deep_ocean(wx, wz)
                except Exception as e:
                    print(f"[WARNING] is_deep_ocean({wx},{wz}) failed: {e}", file=sys.stderr)
                    continue

                if is_deep:
                    chunk_has_deep = True
                    # Req. Sec. 3: fetch the temperature of a deep-sea point when found.
                    # reuse is decided purely by coordinate coincidence (not by whether chunk_temperature is None):
                    # when temperature is off, the chunk center was already billed by _probe_temperature; if we skipped reuse because temp is None,
                    # temp_off would query once more per point than temp_on, breaking fair ablation budgets.
                    if (wx, wz) == (center_x, center_z):
                        temp = chunk_temperature          # already queried at the chunk center; reuse (None when off)
                    else:
                        temp = self._probe_temperature(wx, wz)   # non-center point; billed identically for both groups

                    if self.config["temperature_enabled"]:
                        self.temperature_history.append(temp)
                    found_temperature = temp
                    self._last_temperature_feedback = temp
                    nearest_dist = self.db.get_nearest_distance(wx, wz)

                    # spacing-threshold dedup is handled by the database (Req. Sec. 14)
                    # store 0.0 as a placeholder when temperature is off; never fabricate sensory feedback
                    stored_temp = temp if temp is not None else 0.0
                    if self.db.add_point(wx, wz, stored_temp):
                        is_new_valid_point = True
                        self.deep_sea_points.append((wx, wz, stored_temp))

                        if stage == 3:
                            # Req. Sec. 1 #7: direction & step memory
                            self.agent.record_discovery(action_taken)
                            # long-jump: reset the no-progress counter after a new point is confirmed
                            self._stage3_idle_steps = 0

                        # Req. Sec. 1 #8: data injection for adaptive search granularity + autonomous temperature analysis
                        self._sync_finder_density()
                        self._sync_strategy_reference_temps()

                        # Req. Sec. 10: live query output only in detailed mode
                        if self.is_detailed_mode:
                            temp_str = f"{temp:.4f}" if temp is not None else "N/A"
                            print(f"  [LIVE] deep-sea hit ({wx}, {wz}) temp={temp_str} "
                                  f"total={len(self.deep_sea_points)} "
                                  f"queries={self.cubiomes.query_count}/{max_queries}")
                    # Req. Sec. 2: stop scanning the current chunk as soon as one deep-sea point is found
                    break

            # === record full-grid temperature samples (temp, whether this chunk contains deep sea) ===
            # this is the data basis of the "temperature -> deep-sea probability" statistics: non-deep chunks provide the denominator.
            if self.config["temperature_enabled"] and chunk_temperature is not None:
                self.temperature_samples.append((chunk_temperature, chunk_has_deep))

            # === state transition ===
            # Req. Sec. 15: after finding a deep-sea point, the next state uses the actual sample point's temperature rather than the chunk center's;
            # otherwise the chunk center's temperature serves as the sensory feedback (else temperature sensing cannot enter the state encoding)
            state_temperature = found_temperature if found_temperature is not None else chunk_temperature
            next_state = self.state_manager.get_state(
                *self.current_chunk, temperature=state_temperature
            )

            # curiosity ordering constraint (see state_manager.py): is_new_state must precede mark_visited
            is_new_state = self.state_manager.is_new_state(next_state)

            # === reward computation & Q-learning update (Stage 3 only) ===
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

            # record stage-switch points for convergence analysis
            new_stage = self._get_current_stage()
            if new_stage != stage:
                self._stage_transitions.append({
                    "from_stage": stage,
                    "to_stage": new_stage,
                    "query_count": self.cubiomes.query_count,
                    "deep_sea_points": len(self.deep_sea_points),
                })

        # episode end: apply a final update with done=True for the last step
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
            # Req. Sec. 1 #6: temperature -> deep-sea probability; the agent autonomously identifies the favorable range
            "temperature_deep_probability": self.build_temperature_deep_probability(),
            "temperature_preference": self._analyze_temperature_preference(),
            "temperature_samples": list(self.temperature_samples),
            "stage_transitions": list(self._stage_transitions),
            "visited_states": self.state_manager.visited_count,
            "visited_chunks": len(self._visited_chunks),
            # temperature probe counts: must be strictly equal between temp_on and temp_off,
            # for downstream scripts to verify fair ablation budgets (identical scanned chunk counts)
            "temperature_probes": self._temperature_probe_count,
        }

        # Req. Sec. 10: dual output - detailed mode prints the full result; simple mode prints a one-line summary
        if self.is_detailed_mode:
            print(f"[DETAILED] episode complete | seed={seed} | total queries={self.cubiomes.query_count} | "
                  f"deep-sea points={len(self.deep_sea_points)} | final stage=Stage{stage}")
            print(f"         epsilon={self.agent.epsilon:.4f} | "
                  f"visited states={self.state_manager.visited_count} | "
                  f"scanned chunks={len(self._visited_chunks)}")
            print(f"         deep-sea coords: {[f'({p[0]},{p[1]})' for p in self.deep_sea_points]}")
            if self.config["temperature_enabled"]:
                pref = result["temperature_preference"]
                if pref:
                    print(f"         temperature analysis: chunk samples={pref['sample_count']} "
                          f"(deep {pref['deep_sample_count']}/non-deep {pref['non_deep_sample_count']}) "
                          f"mean temp={pref['mean_temperature']:.4f} "
                          f"range=[{pref['min_temperature']:.4f}, {pref['max_temperature']:.4f}]")
                    print(f"         AI-discovered deep-sea-favorable range: {pref['best_temperature_bin']} "
                          f"(probability {pref['best_bin_probability']:.2%}, "
                          f"{pref['best_bin_deep']}/{pref['best_bin_total']} chunks)")
                    if "deep_mean_temperature" in pref:
                        print(f"         deep chunks mean temp={pref['deep_mean_temperature']:.4f} vs "
                              f"non-deep chunks mean temp={pref['non_deep_mean_temperature']:.4f} "
                              f"(gap {pref['temperature_gap']:.4f})")
                # Req. Sec. 1 #6: draw the "temperature -> deep-sea probability" histogram
                self.print_temperature_deep_probability()
            if self._stage_transitions:
                for tr in self._stage_transitions:
                    print(f"         stage switch: Stage{tr['from_stage']} -> Stage{tr['to_stage']} "
                          f"@queries {tr['query_count']} (deep-sea {tr['deep_sea_points']})")
        else:
            print(f"[SIMPLE] seed={seed} | deep-sea={len(self.deep_sea_points)} | "
                  f"queries={self.cubiomes.query_count} | Stage{stage}")

        return result

    # ==================== Batch experiment entry (Req. Sec. 14) ====================

    def run_experiment(self, world_seeds: List[int], repeats: int,
                       algo_seed_start: int = 42) -> dict:
        """
        Run world_seeds x repeats episodes in batch and aggregate statistics.
        Req. Sec. 12: fixed terrain seeds control environmental variance; repeats measure algorithmic variance.

        Dual output is selected by config["output_mode"] (Req. Sec. 10):
            simple   -> one average row per world seed (20 seeds = 20 rows)
            detailed -> full result per run + live query output

        WARNING (Req. Sec. 15): CSV generation and plotting never happen in this file; this method only returns a statistics dict.

        Args:
            world_seeds: list of world seeds (controls cubiomes terrain generation)
            repeats: number of algorithm repetitions per world seed
            algo_seed_start: starting value of algorithm random seeds (controls internal randomness such as epsilon-greedy)
        Returns:
            statistics dict with mean/std/n_runs/success_rate and all raw results

            success_rate definition = mean deep-sea points / target deep-sea points (completion ratio),
            not the "share of runs that fully reached the target". The completion ratio keeps fine differences between configs in ablations;
            with the target-hit-count definition most configs would all show 0%, losing all discriminative power.
            target_reached_count is kept separately as a reference.
        """
        all_points: List[int] = []
        all_queries: List[int] = []
        raw_results: List[dict] = []
        algo_seed_counter = algo_seed_start
        detailed = self.is_detailed_mode
        target_points = self.config["target_deep_sea_points"]

        if detailed:
            print(f"[DETAILED] batch experiment: {len(world_seeds)} world seeds x {repeats} repeats "
                  f"= {len(world_seeds) * repeats} runs")

        for world_seed in world_seeds:
            seed_points: List[int] = []
            seed_queries: List[int] = []
            seed_target_reached = 0

            for repeat_index in range(repeats):
                # separate world seeds from algorithm seeds (Req. Sec. 12)
                if hasattr(self.cubiomes, "set_seed"):
                    self.cubiomes.set_seed(world_seed)

                algo_seed = algo_seed_counter
                algo_seed_counter += 1

                try:
                    result = self.run_single_episode(seed=algo_seed)
                    points = result["deep_sea_points_found"]
                    queries = result["total_queries"]
                except Exception as e:
                    print(f"\n  [ERROR] world_seed={world_seed} algo_seed={algo_seed} failed: {e}",
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

                # detailed mode: print the full result per run (live query output is already printed inside run_single_episode)
                if detailed:
                    print(f"  [DETAILED] world_seed={world_seed} run {repeat_index + 1}/{repeats} | "
                          f"algo_seed={algo_seed} | deep-sea={points} | queries={queries} | "
                          f"Stage{result.get('final_stage')} | "
                          f"epsilon={result.get('epsilon', float('nan')):.4f}")

            # simple mode: one average row per world seed (20 seeds => 20 rows)
            if not detailed and seed_points:
                avg_p = sum(seed_points) / len(seed_points)
                avg_q = sum(seed_queries) / len(seed_queries)
                # success rate = mean deep-sea points / target deep-sea points
                seed_rate = avg_p / target_points
                print(f"  [SIMPLE] world_seed {world_seed:>2} | avg deep-sea={avg_p:>6.2f} | "
                      f"avg queries={avg_q:>7.1f} | "
                      f"success={avg_p:.2f}/{target_points} ({seed_rate * 100:>5.1f}%)")

        n = len(all_points)
        mean_points = sum(all_points) / n if n else 0.0
        mean_queries = sum(all_queries) / n if n else 0.0
        std_points = math.sqrt(sum((p - mean_points) ** 2 for p in all_points) / n) if n else 0.0
        std_queries = math.sqrt(sum((q - mean_queries) ** 2 for q in all_queries) / n) if n else 0.0

        # success rate = mean deep-sea points / target (completion-ratio definition, for fair cross-ablation comparison)
        success_rate = (mean_points / target_points) if target_points > 0 else 0.0
        # auxiliary: number of runs that fully reached the target (original success_count definition, kept for reference)
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

        print(f"  [SUMMARY] avg deep-sea={mean_points:.2f}±{std_points:.2f} | "
              f"avg queries={mean_queries:.1f}±{std_queries:.1f} | "
              f"success={mean_points:.2f}/{target_points} ({success_rate * 100:.1f}%) | "
              f"runs reaching target={target_reached_count}/{n}")

        return total_stats
