# -*- coding: utf-8 -*-
"""species_agent.py — 黑盒物种搜索框架（用户框架的物种版迁移）
对应 deep-sea-exploration/src/main_algorithm.py 的结构映射：
    cubiomes_wrapper  → oracle.SpeciesOracle（黑盒查询 温度→协变量, 深海→出现）
    database          → oracle 内 presence + 分散约束
    state_manager     → 协变量分箱 + 区域（cov_in_state）
    Stage1/2/3        → 盲搜 → 参数积累 → 参数导向+Q-learning+远跳
    temperature_guidance → 分箱密度导向（用已查询网格统计"参数区间→出现率"）
    Stage3 远跳机制    → 无进展时从已发现点远跳（跳距=间距阈值×ratio，变量绑定）
"""
import math
import random

import numpy as np

from oracle import SpeciesOracle

# ---- 动作空间：8 方向 × 4 步长（网格单位）----
DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
STEPS = [2, 5, 10, 20]  # 网格步长（~5km/13km/26km/52km）
N_BINS = 8


class SpeciesSearcher:
    def __init__(self, oracle: SpeciesOracle, config: dict):
        self.oracle = oracle
        self.cfg = config
        self.target = int(config.get("target_points", 20))
        self.max_queries = int(config.get("max_queries", 20000))
        self.min_dist = float(config.get("min_dist_cells", 1.0))
        self.stage2_min = int(config.get("stage2_min", 3))
        self.stage3_min = int(config.get("stage3_min", 5))
        self.cov_indices = config.get("cov_indices", list(range(oracle.covs.shape[0])))
        self.stage3_jump_interval = int(config.get("stage3_jump_interval", 30))
        self.jump_min_ratio = float(config.get("stage3_jump_min_ratio", 1.25))
        self.jump_max_ratio = float(config.get("stage3_jump_max_ratio", 5.0))
        self.curiosity_weight = float(config.get("curiosity_weight", 1.0))
        self.lr = float(config.get("learning_rate", 0.1))
        self.gamma = float(config.get("discount_factor", 0.95))
        self.eps_start = float(config.get("epsilon_start", 0.9))
        self.eps_end = float(config.get("epsilon_end", 0.1))
        self.eps_decay = float(config.get("epsilon_decay", 0.999))
        self.stage1_grid_step = int(config.get("stage1_grid_step", 2))
        # 改进1：冷启动盲扫铺开。"spread"=象限轮转交错铺开（默认，小预算下空间覆盖
        # 均匀、不取决于起始列）；"serpentine"=完整复现旧蛇形光栅（供 A/B 回归）。
        self.stage1_scan_mode = config.get("stage1_scan_mode", "spread")
        self.n_bins = int(config.get("n_bins", N_BINS))   # 分箱数可配置（消融维度）
        self._rng = random.Random(0)      # episode rng（run_episode 注入，保证可复现）
        self.n_actions = 8 * len(STEPS)
        self.guidance_on = config.get("guidance_on", True)
        self.cluster_guidance = bool(config.get("cluster_guidance", False))
        # ---- 优化项：多簇分别追 + 覆盖预算 + 三级远跳 ----
        # 全部默认开启；传 False / 0 / 0.0 即逐项关闭，关闭后复现旧行为（供 A/B）
        self.multicluster = bool(config.get("multicluster", True))
        self.cluster_ring_ratio = float(config.get("cluster_ring_ratio", 10.0))
        self.cluster_idle_limit = int(config.get("cluster_idle_limit", 12))
        self.blank_jump = bool(config.get("blank_jump", True))
        self.coverage_budget_ratio = float(config.get("coverage_budget_ratio", 0.12))
        # 迭代v2：导向空白跳远距阈值 / 跳后再跳宽限步数 / 饱和簇掩膜开关
        self.blank_far_ratio = float(config.get("blank_far_ratio", 0.3))
        self.post_jump_grace = int(config.get("post_jump_grace", 6))
        self.saturate_mask = bool(config.get("saturate_mask", False))
        # 迭代v2.1：掩膜延迟施加（idle 超过此值才从导向图摘除，给慢热簇留探索时间）
        self.mask_idle_limit = int(config.get("mask_idle_limit", 20))
        self.idle_blank_mix = int(config.get("idle_blank_mix", 8))
        # 改进3：运动外推（默认关闭）。开启后用已发现点时序估计漂移速度矢量，
        # 把预测位置注入 Stage2/3 候选，适配海上搜救类动目标。关闭时与 variant F 逐点一致。
        self.motion_extrapolation = bool(config.get("motion_extrapolation", False))
        # 时间衰减/遗忘机制（四轮）：历史样本随时间淡出，新观测主导导向图。
        # time_decay=False 时与二轮版逐点等价。
        self.time_decay = bool(config.get("time_decay", False))
        self.decay_mode = config.get("decay_mode", "exp")          # "exp" / "window"
        self.decay_tau = float(config.get("decay_tau", 300.0))     # 指数时间常数（查询数）
        self.decay_window = int(config.get("decay_window", 400))    # 滑窗大小
        self.decay_min_weight = float(config.get("decay_min_weight", 0.05))  # AUC 时效过滤阈值
        self.q_decay_tau = float(config.get("q_decay_tau", 200.0)) # Q 表衰减时间常数
        self.q_decay_every = int(config.get("q_decay_every", 25))  # Q 表衰减周期（查询数）
        # 四轮-M1：突变检测+空窗压缩（默认全 False，关闭时与三轮版逐点等价）
        self.mutation_detect = bool(config.get("mutation_detect", False))
        self.mutation_streak_thresh = int(config.get("mutation_streak_thresh", 40))
        self.mutation_accel = float(config.get("mutation_accel", 2.0))
        self.mutation_ramp = int(config.get("mutation_ramp", 40))
        self.mutation_auto_motion = bool(config.get("mutation_auto_motion", False))
        # 四轮-M2：Stage1 先验聚焦（默认 False）
        self.stage1_prior = bool(config.get("stage1_prior", False))
        self.stage1_prior_cov = int(config.get("stage1_prior_cov", 8))
        # 四轮-M3：高维弱信号加权（默认 False）
        self.weak_signal_boost = bool(config.get("weak_signal_boost", False))
        self.weak_signal_min_auc = float(config.get("weak_signal_min_auc", 0.52))
        # 五轮-M4：跟随模式（默认 False；关闭时与四轮逐位等价）
        self.follow_mode = bool(config.get("follow_mode", False))
        self.follow_window = int(config.get("follow_window", 24))
        self.follow_radius = float(config.get("follow_radius", 8.0))
        # 五轮-M6：突变旱期撒网跳（默认 False）
        self.mutation_disperse_jump = bool(config.get("mutation_disperse_jump", False))
        self.mutation_jump_interval = int(config.get("mutation_jump_interval", 15))
        # 每协变量全网格 min/max（用于状态分箱，自适应）
        self._cov_range = []
        for k in range(oracle.covs.shape[0]):
            v = oracle.covs[k]
            self._cov_range.append((float(np.nanmin(v)), float(np.nanmax(v))))
        self.reset()

    def reset(self):
        self.found_cells = []
        self.visited_cells = set()
        self.stage = 1
        self.q = {}
        self.eps = self.eps_start
        # 数据积累：已查询网格的 (协变量向量, 是否出现) 样本
        self._queries = []          # [(cov_vec, presence_bool)]
        self._feature_log = []      # 特征选择诊断：(n_q, n_pos, [(cov_idx, weight)...])
        self._densities = None      # {k: (edges, density)}
        self._weights = {}          # {k: 区分度权重}
        self._score_map = None      # 全网格导向分
        self._stage3_idle = 0
        self.cur_cell = None
        self.convergence = []
        # 多簇簿记（优化项1）：[{members, centroid, cover_radius, last_grown_q, saturated}]
        self._clusters = []
        self._last_found_cell = None      # 本轮新发现点（簇生长记账用）
        self.stage1_end_coverage = None   # Stage1 结束时覆盖比例（优化项2 显式记录）
        # 迭代v2：跳后节奏 / 饱和簇掩膜缓存
        self._steps_since_jump = 0
        self._has_jumped = False
        self._far_cooldown = False
        self._eff_sm = None
        self._eff_sig = None
        # 改进1：spread 盲扫游标（交错列序光栅；首次进入 Stage1 时惰性初始化）
        self._s1_j0 = None
        self._s1_band_w = 0
        self._s1_B = 0
        self._s1_offset = 0
        self._s1_band = 0
        self._s1_row = 0
        # 改进3：已发现点的时间戳（=发现时 query_count），与 found_cells 平行簿记
        self._found_t = []
        # 四轮：与 _queries 平行的时间戳（记录时样本数），仅 time_decay 分支读取
        self._query_t = []
        # 四轮-M1：突变检测簿记（最后一次命中时的 query_count）
        self._last_hit_q = -1
        # 五轮-M4：跟随模式簿记（最后一次命中时的 query_count，无 mutation_detect 门控）
        self._last_hit_q_any = -1
        # 四轮-M2：Stage1 先验聚焦簿记（惰性构建）
        self._s1_prior_order = None
        self._s1_prior_pos = 0

    # ---- 分散约束 ----
    def _distance_ok(self, i, j):
        if not self.found_cells:
            return True
        for fi, fj in self.found_cells:
            if math.hypot(i - fi, j - fj) < self.min_dist:
                return False
        return True

    # ---- 分箱密度导向（用全部已查询正负样本；按变量区分度加权）----
    # ---- 区分度精筛：Mann-Whitney AUC（噪声变量≈0.5，对小样本波动免疫）----
    def _auc(self, pos_vals, neg_vals):
        n, m = len(pos_vals), len(neg_vals)
        if n == 0 or m == 0:
            return 0.5
        # 抽样控制计算量（用 episode rng，保证可复现）
        if n > 200:
            pos_vals = self._rng.sample(list(pos_vals), 200)
            n = 200
        if m > 800:
            neg_vals = self._rng.sample(list(neg_vals), 800)
            m = 800
        x = np.concatenate([pos_vals, neg_vals])
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty(len(x), dtype=np.float64)
        ranks[order] = np.arange(1, len(x) + 1)
        xs = x[order]
        i, L = 0, len(xs)
        while i < L:                       # ties 平均秩
            j = i
            while j + 1 < L and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        rsum = float(ranks[:n].sum())
        u = rsum - n * (n + 1) / 2.0
        auc = u / (n * m)
        return max(auc, 1.0 - auc)        # 方向无关，只取分离度

    # ---- 四轮：时间衰减/遗忘机制 ----
    def _decay_weights(self, accel=True):
        """返回长度 n 的 float 权重向量（样本 t 的年龄 = n-1 - _query_t[t]）。
        time_decay=False 返回全 1；window 模式超窗权重 0；exp 模式指数衰减。
        accel=False 时跳过 M1 突变加速（供 AUC 选参/n_pos 门限解耦用）。"""
        n = len(self._queries)
        if not self.time_decay or n == 0:
            return np.ones(n, dtype=np.float64)
        ages = (n - 1) - np.asarray(self._query_t, dtype=np.float64)
        # 四轮-M1：突变旱期旧样本有效年龄放大（最多 ×(1+mutation_accel)），加速淡出
        if self.mutation_detect and accel:
            streak = self._mutation_streak()
            if streak > self.mutation_streak_thresh:
                over = min(1.0, (streak - self.mutation_streak_thresh) / max(self.mutation_ramp, 1))
                ages = ages * (1.0 + self.mutation_accel * over)
        if self.decay_mode == "window":
            w = np.where(ages >= self.decay_window, 0.0, 1.0)
            if self.decay_window <= 0:
                w = np.ones(n, dtype=np.float64)
            return w
        else:  # "exp"
            tau = max(self.decay_tau, 1e-9)
            if self.decay_tau >= 1e7:
                return np.ones(n, dtype=np.float64)
            return np.exp(-ages / tau)

    # ---- 四轮-M1：突变检测 helper ----
    def _mutation_streak(self):
        """当前"全局零命中"查询数（样本索引域：n-1 - last_hit_q）。"""
        if not self.mutation_detect:
            return 0
        n = len(self._queries)
        if self._last_hit_q < 0:
            return n - 1 if n > 0 else 0
        return max((n - 1) - self._last_hit_q, 0)

    # ---- 五轮-M4：跟随模式 helpers ----
    def _follow_active(self):
        """跟随模式激活：follow_mode=True 且最近一次命中在 follow_window 查询内。
        OFF 时短路返回 False（不读任何新簿记，逐位等价）。"""
        if not self.follow_mode or self._last_hit_q_any < 0:
            return False
        return (self.oracle.query_count - self._last_hit_q_any) < self.follow_window

    def _follow_ring(self, k):
        """锚定最近发现点的 [min_dist, follow_radius] 环带内未访问格（随机序）。
        环空返回 []（调用方立即回退原路径）。仅 follow_mode ON 时被调用。"""
        if self._last_found_cell is None:
            return []
        o = self.oracle
        cx, cy = self._last_found_cell
        r = int(self.follow_radius)
        min_d = max(int(self.min_dist), 1)
        cand = []
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                d = math.hypot(di, dj)
                if d < min_d or d > r:
                    continue
                ni, nj = cx + di, cy + dj
                if o.in_bounds(ni, nj) and (ni, nj) not in self.visited_cells:
                    cand.append((ni, nj))
        if not cand:
            return []
        self._rng.shuffle(cand)
        return cand[:k]

    # ---- 五轮-M6：突变旱期撒网跳 ----
    def _mutation_disperse_jump(self):
        """突变窗口内：距所有已发现点 >= far_dist 的未访问粗网格格均匀随机取一个。
        刻意不用导向分（突变后导向图仍指旧区，导向跳会被拉回旧区）。"""
        o = self.oracle
        step = max(8, min(o.nrows, o.ncols) // 8)
        far_dist = self.blank_far_ratio * min(o.nrows, o.ncols)
        pool = []
        for i in range(0, o.nrows, step):
            for j in range(0, o.ncols, step):
                if (i, j) in self.visited_cells:
                    continue
                dmin = min(math.hypot(i - fi, j - fj)
                           for fi, fj in self.found_cells) if self.found_cells else 1e9
                if dmin >= far_dist:
                    pool.append((i, j))
        if not pool:
            return False
        self._rng.shuffle(pool)
        self.cur_cell = pool[0]
        return True

    def _auc_decayed(self, pos_vals, neg_vals, w_pos, w_neg):
        """按 权重 >= decay_min_weight 过滤后调用原 _auc。
        任一侧过滤后为空返回 0.5。"""
        keep_p = w_pos >= self.decay_min_weight
        keep_n = w_neg >= self.decay_min_weight
        pv = pos_vals[keep_p]
        nv = neg_vals[keep_n]
        if len(pv) == 0 or len(nv) == 0:
            return 0.5
        return self._auc(pv, nv)

    def _decay_q(self):
        """把所有 q[s][a] 乘以指数衰减因子。纯算术，不消耗 self._rng。
        q_decay_tau>=1e7 视为无穷（无衰减效果，直接返回以保证数值等价）。"""
        if self.q_decay_tau <= 0 or self.q_decay_tau >= 1e7:
            return
        factor = math.exp(-self.q_decay_every / max(self.q_decay_tau, 1e-9))
        for s in self.q:
            for a in self.q[s]:
                self.q[s][a] *= factor

    def _build_density(self):
        """对每个协变量分 N_BINS 箱，density = 箱内出现查询数/箱内查询数；
        同时计算各变量分离度权重，区分度高的变量权重大。
        v2 稀疏化（sparse_guidance=True）：
          正样本足够后，先按箱密度方差粗筛 top-12，再以 AUC 精筛，
          只保留统计上区分度显著的变量（AUC>0.55），权重按 (AUC-0.5)^2 幂律压缩。
          topK 随正样本数渐进放宽：样本越多，估计越可靠，可保留越多有效变量。
          噪声变量 AUC≈0.5 被天然排除 → 参数多 = 候选池大，在线选择器自动挑出有效子集。"""
        if len(self._queries) < self.stage2_min:
            return
        covs = np.array([q[0] for q in self._queries])       # (n, K)
        pres = np.array([q[1] for q in self._queries], dtype=float)
        # 四轮：时间衰减权重向量（time_decay=False 时全 1）
        w = self._decay_weights()                  # 密度导向：突变期加速
        w_sel = self._decay_weights(accel=False)   # 四轮-M1解耦：AUC选参/n_pos门限用不加速权重
        self._densities = {}
        raw_w = {}
        for k in self.cov_indices:
            v = covs[:, k]
            lo, hi = float(np.min(v)), float(np.max(v))
            if hi - lo < 1e-9:
                hi = lo + 1.0
            edges = np.linspace(lo, hi, self.n_bins + 1)
            b = np.clip(np.digitize(v, edges) - 1, 0, self.n_bins - 1)
            dens = np.zeros(self.n_bins)
            if self.time_decay:
                cnt = np.bincount(b, weights=w, minlength=self.n_bins)
                pos = np.bincount(b, weights=pres * w, minlength=self.n_bins)
            else:
                cnt = np.bincount(b, minlength=self.n_bins)
                pos = np.bincount(b, weights=pres, minlength=self.n_bins)
            nz = cnt > 0
            dens[nz] = pos[nz] / cnt[nz]
            # 平滑：无查询箱用全局出现率
            dens[~nz] = pos.sum() / max(cnt.sum(), 1)
            self._densities[k] = (edges, dens)
            raw_w[k] = float(np.var(dens))   # 粗筛：箱密度方差

        # ---- v2 稀疏化：AUC 精筛 + 幂律压缩 + topK 渐进 ----
        if self.cfg.get("sparse_guidance", False):
            if self.time_decay:
                n_pos = int(round(float(np.sum(pres * w_sel))))
            else:
                n_pos = int(pres.sum())
            n_q = len(self._queries)
            # v3 冷启动（默认关闭，保持 v2 可复现）：总样本积累不足时全量加权
            # （AUC 在小样本下噪声大），过阈值后才启用稀疏；仅在实验脚本显式
            # 传入 cold_start_samples>0 时生效
            cold_start = int(self.cfg.get("cold_start_samples", 0))
            if n_pos >= self.cfg.get("sparse_min_pos", 5) and n_q >= cold_start:
                topk_base = int(self.cfg.get("sparse_topk", 6))
                n_vars = len(raw_w)
                # 正样本越多，区分度估计越可靠 → 可保留更多有效变量
                topk = min(topk_base + max(0, (n_pos - 5) // 15), n_vars)
                # 1) 方差粗筛 top-12（快），避免对 19 变量全量算 AUC
                ranked_v = sorted(raw_w.items(), key=lambda x: -x[1])[:12]
                # 2) AUC 精筛：噪声变量≈0.5，真信号稳定>0.55
                pos_idx = np.where(pres > 0.5)[0]
                neg_idx = np.where(pres < 0.5)[0]
                auc_w = {}
                for k, _ in ranked_v:
                    if self.time_decay:
                        auc_w[k] = self._auc_decayed(
                            covs[pos_idx, k], covs[neg_idx, k],
                            w_sel[pos_idx], w_sel[neg_idx])
                    else:
                        auc_w[k] = self._auc(covs[pos_idx, k], covs[neg_idx, k])
                ranked_a = sorted(auc_w.items(), key=lambda x: -x[1])[:topk]
                # 3) 只保留显著变量，权重 = (AUC-0.5)^2 * 4（软稀疏：弱信号近清零）
                if self.weak_signal_boost:
                    keep = {}
                    for k, w in ranked_a:
                        if w > 0.55:
                            keep[k] = max(w - 0.5, 0.0) ** 2 * 4.0
                        elif w > self.weak_signal_min_auc:
                            keep[k] = max(w - 0.5, 0.0) * 2.0
                else:
                    keep = {k: max(w - 0.5, 0.0) ** 2 * 4.0 for k, w in ranked_a if w > 0.55}
                if keep:
                    raw_w = keep
                # 保守回退：显著变量为空（样本极端）则保留 top1，避免导向全失
                # 特征选择诊断：记录本轮选中的代表特征及其权重（按权重降序）
                self._feature_log.append(
                    (len(self._queries), n_pos,
                     sorted(keep.items(), key=lambda x: -x[1])))

        tot = sum(raw_w.values())
        self._weights = {k: (w / tot if tot > 0 else 1.0 / len(raw_w))
                         for k, w in raw_w.items()}

    def _cell_score(self, i, j):
        """网格导向分：协变量所在箱的出现密度，按变量区分度加权"""
        if self._densities is None:
            return 0.0
        c = self.oracle.covs[:, i, j]
        s = 0.0
        for k in self.cov_indices:
            edges, dens = self._densities[k]
            v = float(c[k])
            b = np.clip(np.digitize(v, edges) - 1, 0, N_BINS - 1)
            s += self._weights.get(k, 0.0) * dens[b]
        return s

    def _build_score_map(self):
        o = self.oracle
        sm = np.zeros((o.nrows, o.ncols), dtype=np.float32)
        if self._densities is None:
            self._score_map = sm
            return
        for k in self.cov_indices:
            edges, dens = self._densities[k]
            b = np.clip(np.digitize(o.covs[k], edges) - 1, 0, self.n_bins - 1)
            sm += self._weights.get(k, 0.0) * dens[b]
        self._score_map = sm

    def _record_query(self, i, j, pres):
        self._queries.append((self.oracle.covs[:, i, j].copy(), bool(pres)))
        self._query_t.append(len(self._queries) - 1)  # 时间戳 = 样本索引（与最新样本的距离即 age）
        # 密度表/导向图每 recalc_every 次查询重建一次（避免 O(K×N) 全量重算拖慢 19 参数）
        n_q = len(self._queries)
        if self.stage >= 2 and (n_q % self.cfg.get("recalc_every", 20) == 0
                                or n_q == self.stage2_min):
            self._build_density()
            self._build_score_map()
        # 四轮：Q 表周期性衰减（密度重建块之后）
        if (self.time_decay and self.q_decay_tau > 0
                and n_q % self.q_decay_every == 0):
            self._decay_q()

    # ---- 聚类感知导向（多模态：候选按集群分散，避免全局导向锁死单簇）----
    def _cluster_found_cells(self):
        """分箱网格连通聚类（8 邻域 BFS）：把已发现点分成空间集群"""
        o = self.oracle
        n = len(self.found_cells)
        if n < 3:
            return [list(self.found_cells)]
        bs_i = max(1, o.nrows // self.n_bins)
        bs_j = max(1, o.ncols // self.n_bins)
        bins = {}
        order = []
        for (i, j) in self.found_cells:
            b = (i // bs_i, j // bs_j)
            if b not in bins:
                bins[b] = []
                order.append(b)
            bins[b].append((i, j))
        seen = set()
        clusters = []
        for b0 in order:
            if b0 in seen:
                continue
            stack = [b0]
            seen.add(b0)
            members = []
            while stack:
                b = stack.pop()
                members.extend(bins.get(b, []))
                bi, bj = b
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        nb = (bi + di, bj + dj)
                        if nb in bins and nb not in seen:
                            seen.add(nb)
                            stack.append(nb)
            clusters.append(members)
        return clusters

    def _cluster_ring_candidates(self, cluster, min_d, max_d):
        """簇热区：以簇成员为中心、[min_d, max_d] 环带内未访问网格，按全局导向分降序去重"""
        o = self.oracle
        sm = self._score_map if self._score_map is not None else None
        cand = []
        for (fi, fj) in cluster:
            for di in range(-max_d, max_d + 1):
                for dj in range(-max_d, max_d + 1):
                    d = math.hypot(di, dj)
                    if d < min_d or d > max_d:
                        continue
                    ni, nj = fi + di, fj + dj
                    if not o.in_bounds(ni, nj) or (ni, nj) in self.visited_cells:
                        continue
                    s = float(sm[ni, nj]) if sm is not None else 0.0
                    cand.append((s, ni, nj))
        seen = set()
        uniq = []
        for s, ni, nj in sorted(cand, key=lambda x: -x[0]):
            if (ni, nj) in seen:
                continue
            seen.add((ni, nj))
            uniq.append((ni, nj))
        return uniq

    # ---- 优化项1：多簇簿记（簇划分复用 _cluster_found_cells 分箱 8 邻域 BFS）----
    def _sync_clusters(self):
        """依据 found_cells 重建簇划分，并沿用上一轮簇的生长时间/饱和状态。
        每簇 dict: {members, centroid, cover_radius, last_grown_q, saturated}。
        新点归入：由分箱 BFS 自然落簇；本轮新发现点所在簇 last_grown_q 刷新、
        饱和解除；连续无新点查询数 > cluster_idle_limit → 标记饱和。"""
        if not self.found_cells:
            self._clusters = []
            return
        raw = self._cluster_found_cells()
        prev = self._clusters
        out = []
        for members in raw:
            cx = sum(float(m[0]) for m in members) / len(members)
            cy = sum(float(m[1]) for m in members) / len(members)
            cover = max(math.hypot(m[0] - cx, m[1] - cy) for m in members)
            best, bd = None, None
            for pc in prev:
                d = math.hypot(pc["centroid"][0] - cx, pc["centroid"][1] - cy)
                if bd is None or d < bd:
                    best, bd = pc, d
            inherit = best is not None and bd <= cover + self.min_dist + 2
            last_q = best["last_grown_q"] if inherit else 0
            sat = best["saturated"] if inherit else False
            if self._last_found_cell in members:
                last_q = self.oracle.query_count
                sat = False
            out.append({"members": members, "centroid": (cx, cy),
                        "cover_radius": cover, "last_grown_q": last_q,
                        "saturated": sat})
        for cl in out:
            if self.oracle.query_count - cl["last_grown_q"] > self.cluster_idle_limit:
                cl["saturated"] = True
        self._clusters = out

    def _ring_around_centroid(self, centroid, min_d, max_d):
        """以簇质心为中心、[min_d, max_d] 环带内未访问网格，按全局导向分降序去重。
        半径上界 = cluster_ring_ratio × min_dist（替换旧 max(5, min_dist×jump_max_ratio)
        的近视力）；单中心扫描替代逐成员循环，省算力。"""
        o = self.oracle
        sm = self._score_map
        cx, cy = int(round(centroid[0])), int(round(centroid[1]))
        r = int(max_d)
        cand = []
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                d = math.hypot(di, dj)
                if d < min_d or d > max_d:
                    continue
                ni, nj = cx + di, cy + dj
                if not o.in_bounds(ni, nj) or (ni, nj) in self.visited_cells:
                    continue
                s = float(sm[ni, nj]) if sm is not None else 0.0
                cand.append((s, ni, nj))
        cand.sort(key=lambda x: -x[0])
        return [(i, j) for _, i, j in cand]

    def _effective_score_map(self):
        """迭代v2-B：长时间无新点（idle > mask_idle_limit）的簇 cover 区域从导向分中
        临时置零，防 Q/导向把搜索拉回已采尽簇；给慢热簇留 mask_idle_limit 步探索期。
        无待掩膜簇 / 全部簇待掩膜 / multicluster=False → 原样返回 score_map。"""
        if not self.multicluster or not self.saturate_mask or self._score_map is None:
            return self._score_map
        self._sync_clusters()
        stale = [c for c in self._clusters
                 if self.oracle.query_count - c["last_grown_q"] > self.mask_idle_limit]
        if not stale or len(stale) == len(self._clusters):
            return self._score_map
        sig = (id(self._score_map), self.mask_idle_limit,
               tuple(sorted((round(c["centroid"][0], 1), round(c["centroid"][1], 1),
                             round(c["cover_radius"], 1)) for c in stale)))
        if self._eff_sig == sig and self._eff_sm is not None:
            return self._eff_sm
        o = self.oracle
        sm = np.array(self._score_map, copy=True)
        for c in stale:
            cx, cy = int(round(c["centroid"][0])), int(round(c["centroid"][1]))
            R = int(c["cover_radius"]) + 2
            for di in range(-R, R + 1):
                for dj in range(-R, R + 1):
                    ni, nj = cx + di, cy + dj
                    if o.in_bounds(ni, nj):
                        sm[ni, nj] = 0.0
        self._eff_sm = sm
        self._eff_sig = sig
        return sm

    def _blank_candidates(self, n=1):
        """迭代v2-A：导向空白跳候选。粗网格步长 max(8, min(nrows,ncols)//8)，
        先过滤'距最近已发现点 ≥ far_dist（=blank_far_ratio×min(nrows,ncols)）'的
        远距未访问格，再按全局导向分降序取 top-8 随机 n 个——跳向
        '远且像目标栖息地'的区域，而非纯随机死区。导向图未建时回退最远采样。"""
        o = self.oracle
        step = max(8, min(o.nrows, o.ncols) // 8)
        far_dist = self.blank_far_ratio * min(o.nrows, o.ncols)
        sm = self._effective_score_map()
        far = []
        near = []
        for i in range(0, o.nrows, step):
            for j in range(0, o.ncols, step):
                if (i, j) in self.visited_cells:
                    continue
                dmin = min(math.hypot(i - fi, j - fj)
                           for fi, fj in self.found_cells) if self.found_cells else 1e9
                s = float(sm[i, j]) if sm is not None else 0.0
                if dmin >= far_dist:
                    far.append((s, i, j))
                else:
                    near.append((dmin, i, j))
        pool = far if far else near
        if not pool:
            return []
        if sm is not None and far:
            # 远距候选按导向分降序
            pool.sort(key=lambda x: -x[0])
            top = pool[:8]
        else:
            # 无导向图或无远距候选：回退最远采样
            pool.sort(key=lambda x: -x[0])
            top = pool[:8]
        self._rng.shuffle(top)
        return [(i, j) for _, i, j in top[:n]]

    def _blank_jump(self):
        """三级远跳 L3：空白区开簇跳。跳距由域尺寸/簇间距决定，不受
        [jump_min_ratio, jump_max_ratio] 比例限制。"""
        picks = self._blank_candidates(1)
        if not picks:
            return False
        self.cur_cell = picks[0]
        return True

    def _cross_cluster_jump(self):
        """三级远跳 L2：跨簇转移跳——跳到另一未饱和簇质心外环的随机候选。"""
        self._sync_clusters()
        unsat = [cl for cl in self._clusters if not cl["saturated"]]
        if len(unsat) < 2:
            return False
        cl = self._rng.choice(unsat)
        min_d = max(int(self.min_dist), 1)
        max_d = self.cluster_ring_ratio * self.min_dist
        ring = self._ring_around_centroid(cl["centroid"], min_d, max_d)
        if not ring:
            return False
        self.cur_cell = self._rng.choice(ring[:12])
        return True

    def _coverage_ok(self):
        """优化项2：Stage1 盲扫预算下限。coverage_budget_ratio<=0 关闭（复旧行为）"""
        r = self.coverage_budget_ratio
        if r <= 0:
            return True
        return self.oracle.query_count >= int(self.max_queries * r)

    # ---- 改进1：冷启动盲扫铺开（spread，交错列序光栅）----
    def _stage1_spread_next(self, i, j, o):
        """以随机起始列 j 为锚，按"交错列序"访问列：先取 j0, j0+band_w, ...,
        j0+(B-1)*band_w（B 个均匀散布列），再逐偏移补齐 j0+1, j0+1+band_w, ...。
        每列内部按 stage1_grid_step 向下扫行；一列扫完才推进到下一交错列。
        小预算下首批查询即散布全列轴（不取决于起始列），又不会像固定锚定列那样
        原地重扫卡死。纯算术推进，不消耗 episode rng。"""
        st = self.stage1_grid_step
        if self._s1_j0 is None:
            B = int(self.cfg.get("stage1_spread_bands", max(4, o.ncols // 10)))
            B = max(2, min(B, o.ncols))
            self._s1_j0 = j
            self._s1_band_w = max(1, o.ncols // B)
            self._s1_B = B
            self._s1_offset = 0
            self._s1_band = 0
            self._s1_row = i + st
        else:
            self._s1_row += st
        if self._s1_row >= o.nrows:
            self._s1_row = st
            self._s1_band += 1
            if self._s1_band >= self._s1_B:
                self._s1_band = 0
                self._s1_offset += 1
                if self._s1_offset >= self._s1_band_w:
                    self._s1_offset = 0
        c = (self._s1_j0 + self._s1_band * self._s1_band_w + self._s1_offset) % o.ncols
        return (self._s1_row, c)

    # ---- 四轮-M2：Stage1 先验聚焦 ----
    def _stage1_prior_next(self, o):
        """按 covs[stage1_prior_cov] 降序返回下一个未访问格；全部访问完返回 None。
        确定性（不消耗 RNG）；首次调用时构建一次排序。"""
        if self._s1_prior_order is None:
            c = self.oracle.covs[self.stage1_prior_cov]
            flat = np.argsort(-c.ravel(), kind="mergesort")
            self._s1_prior_order = [(int(idx // o.ncols), int(idx % o.ncols)) for idx in flat]
            self._s1_prior_pos = 0
        while self._s1_prior_pos < len(self._s1_prior_order):
            cell = self._s1_prior_order[self._s1_prior_pos]
            self._s1_prior_pos += 1
            if cell not in self.visited_cells:
                return cell
        return None

    # ---- 改进3：运动外推 ----
    def _motion_predicted_cell(self):
        """用已发现点时序估计漂移速度矢量（相邻发现位移/时间增量取中位数），
        外推目标当前位置。静态 oracle 下位移≈0 → 预测≈最后发现点（已 visited，
        不会重复注入），模块自然无效。返回 (pi,pj) 或 None。"""
        if not (self.motion_extrapolation
                or (self.mutation_detect and self.mutation_auto_motion
                    and self._mutation_streak() > self.mutation_streak_thresh)):
            return None
        cells = self.found_cells
        ts = self._found_t
        if len(cells) < 2 or len(ts) < 2:
            return None
        disps = []
        for k in range(1, len(cells)):
            di = cells[k][0] - cells[k - 1][0]
            dj = cells[k][1] - cells[k - 1][1]
            dt = max(ts[k] - ts[k - 1], 1)
            disps.append((di / dt, dj / dt))
        if not disps:
            return None
        v_i = sorted(d[0] for d in disps)[len(disps) // 2]
        v_j = sorted(d[1] for d in disps)[len(disps) // 2]
        li, lj = cells[-1]
        dt_now = max(self.oracle.query_count - ts[-1], 1)
        pi = int(round(li + v_i * dt_now))
        pj = int(round(lj + v_j * dt_now))
        if self.oracle.in_bounds(pi, pj):
            return (pi, pj)
        return None

    # ---- 自适应开关（A11）：在线诊断已发现点的空间结构 ----
    # 多集群(≥2 簇) → 聚类分散导向（防全局导向锁死单侧高分簇）；
    # 单集群(1 簇) → 全局导向（协变量单调驱动，导向有效）。
    # adaptive=True 时无需人工指定 cluster_guidance，机器自动切换。
    def _use_cluster_mode(self):
        if not self.cfg.get("adaptive", False):
            return self.cluster_guidance
        clusters = self._cluster_found_cells()
        return len(clusters) >= 2

    # ---- 候选网格 ----
    def _top_unvisited(self, k: int):
        o = self.oracle
        sm = self._effective_score_map() if self._score_map is not None else np.zeros((o.nrows, o.ncols))
        # ---- 优化项1：多簇分别追（multicluster=True 时优先；关闭则完全走旧逻辑）----
        if self.multicluster and self.found_cells:
            self._sync_clusters()
            if len(self._clusters) >= 2:
                active = [cl for cl in self._clusters if not cl["saturated"]]
                if active:
                    min_d = max(int(self.min_dist), 1)
                    max_d = self.cluster_ring_ratio * self.min_dist
                    cands = []
                    per = max(1, k // len(active))
                    for cl in active:
                        ring = self._ring_around_centroid(cl["centroid"], min_d, max_d)
                        cands.extend(ring[:per])
                    seen = set()
                    out = []
                    for c in cands:
                        if c in seen:
                            continue
                        seen.add(c)
                        out.append(c)
                        if len(out) >= k:
                            break
                    if out:
                        return out
                # 兜底：全部簇饱和 → 落到下方旧全局导向
            else:
                # v2f 单簇：走旧全局导向 top-K（CHL 类单一密集热点簇）；
                # 仅当该簇停滞 idle>=idle_blank_mix 时追加空白候选撒网开新簇
                flat = np.argsort(sm.ravel())[::-1]
                out = []
                for idx in flat:
                    i, j = idx // o.ncols, idx % o.ncols
                    if (i, j) not in self.visited_cells:
                        out.append((i, j))
                        if len(out) >= k:
                            break
                if out:
                    idle = (self.oracle.query_count - self._clusters[0]["last_grown_q"]
                            if self._clusters else 1e9)
                    if self.idle_blank_mix > 0 and idle >= self.idle_blank_mix:
                        out.extend(self._blank_candidates(2))
                    return out
        if self._use_cluster_mode():
            clusters = self._cluster_found_cells()
            if len(clusters) >= 2:
                min_d = max(int(self.min_dist), 1)
                max_d = max(5, int(self.min_dist * self.jump_max_ratio))
                cands = []
                per = max(1, k // len(clusters))
                for cl in clusters:
                    ring = self._cluster_ring_candidates(cl, min_d, max_d)
                    cands.extend(ring[:per])
                if len(cands) < k:
                    flat = np.argsort(sm.ravel())[::-1]
                    for idx in flat:
                        i, j = idx // o.ncols, idx % o.ncols
                        if (i, j) not in self.visited_cells and (i, j) not in cands:
                            cands.append((i, j))
                            if len(cands) >= k:
                                break
                seen = set()
                out = []
                for c in cands:
                    if c not in seen:
                        seen.add(c)
                        out.append(c)
                return out
        flat = np.argsort(sm.ravel())[::-1]
        cands = []
        for idx in flat:
            i, j = idx // o.ncols, idx % o.ncols
            if (i, j) not in self.visited_cells:
                cands.append((i, j))
                if len(cands) >= k:
                    break
        return cands

    def _random_unvisited(self):
        o = self.oracle
        for _ in range(300):
            i = self._rng.randrange(o.nrows)
            j = self._rng.randrange(o.ncols)
            if (i, j) not in self.visited_cells:
                return i, j
        return self._rng.randrange(o.nrows), self._rng.randrange(o.ncols)

    # ---- Stage3 远跳（变量绑定跳距：阈值×ratio；定向到未覆盖高分带，提升跨集群覆盖）----
    # v3 定向远跳：从全局导向分前 20% 的未访问网格中，找"距最近已发现点恰好在跳距范围"
    # 的高分目标带（跨集群直达）；高分带不足时回退 8 方向定向扫描 + 评分排序。
    # 保留随机性（top-N 中随机），避免锁定单个高分簇。
    def _far_jump(self):
        if not self.found_cells:
            return False
        # 五轮-M6：突变窗口内撒网跳优先（分散找新区，绕开旧导向偏置）
        if (self.mutation_detect and self.mutation_disperse_jump
                and self._mutation_streak() > self.mutation_streak_thresh):
            if self._mutation_disperse_jump():
                return True
        # ---- 优化项3：三级远跳（L2 跨簇转移 / L3 空白开簇优先；失败回退普通远跳）----
        if self.multicluster:
            self._sync_clusters()
            has_sat = any(cl["saturated"] for cl in self._clusters)
            # 迭代v2-C：跳后 post_jump_grace 步无发现 = 落入死区，同样触发换区
            dead_zone = self._steps_since_jump >= self.post_jump_grace
            if self.blank_jump and (has_sat or dead_zone):
                if self._blank_jump():
                    return True
            if has_sat or dead_zone or self._stage3_idle >= self.stage3_jump_interval:
                if self._cross_cluster_jump():
                    return True
        min_d = max(int(self.min_dist * self.jump_min_ratio), int(self.min_dist) + 2)
        max_d = int(self.min_dist * self.jump_max_ratio)
        o = self.oracle
        cands = []
        if self.cfg.get("directed_jump", False) and self._score_map is not None:
            sm = self._effective_score_map()
            if self._use_cluster_mode():
                # 聚类感知远跳：每簇热区均匀取候选（跨簇直达，避免锁单侧高分簇）
                clusters = self._cluster_found_cells()
                for cl in clusters:
                    ring = self._cluster_ring_candidates(cl, min_d, max_d)
                    for (ni, nj) in ring[:8]:
                        cands.append((float(sm[ni, nj]), ni, nj))
                        if len(cands) >= 24:
                            break
                    if len(cands) >= 24:
                        break
            else:
                thr = float(np.percentile(sm, 80))
                # 全局高分未访问网格（候选目标带）
                hi = np.argwhere(sm.ravel() >= thr)
                hi_list = list(hi)
                self._rng.shuffle(hi_list)
                for idx in hi_list:
                    i, j = int(idx[0] // o.ncols), int(idx[0] % o.ncols)
                    if (i, j) in self.visited_cells:
                        continue
                    dmin = min(math.hypot(i - fi, j - fj) for fi, fj in self.found_cells)
                    if min_d <= dmin <= max_d:
                        cands.append((float(sm[i, j]), i, j))
                        if len(cands) >= 24:
                            break
        if not cands:
            # 回退：从已发现点向 8 方向做"方向内最优距离"定向扫描
            for fi, fj in self._rng.sample(self.found_cells, min(4, len(self.found_cells))):
                for di, dj in DIRS:
                    best = None
                    for d in range(min_d, max_d + 1, max(1, (max_d - min_d) // 6)):
                        ni, nj = fi + di * d, fj + dj * d
                        if o.in_bounds(ni, nj) and (ni, nj) not in self.visited_cells:
                            s = float(self._score_map[ni, nj]) if self._score_map is not None else 0.0
                            if best is None or s > best[0]:
                                best = (s, ni, nj)
                    if best:
                        cands.append(best)
        if not cands:
            return False
        # 导向分高的候选优先（带随机性，避免锁定）
        cands.sort(reverse=True)
        top = cands[: min(6, len(cands))]
        _, ni, nj = self._rng.choice(top)
        self.cur_cell = (ni, nj)
        return True

    # ---- Q-learning ----
    def _state_key(self, i, j):
        if not self.cfg.get("cov_in_state", True):
            return (i // 8, j // 8)
        c = self.oracle.covs[:, i, j]
        bins = []
        for k in self.cov_indices:
            lo, hi = self._cov_range[k]
            if hi - lo < 1e-9:
                b = 0
            else:
                b = int((float(c[k]) - lo) / (hi - lo) * 10)
                b = min(9, max(0, b))
            bins.append(b)
        return (i // 8, j // 8, tuple(bins))

    def _best_guided_action(self, i, j):
        sm = self._effective_score_map()
        best_a, best_s = None, -1.0
        for a in range(self.n_actions):
            di, dj = DIRS[a // len(STEPS)]
            st = STEPS[a % len(STEPS)]
            ni, nj = i + di * st, j + dj * st
            if not self.oracle.in_bounds(ni, nj):
                continue
            s = float(sm[ni, nj]) if sm is not None else 0.0
            if s > best_s:
                best_s, best_a = s, a
        return best_a

    def _choose_action(self, i, j):
        s = self._state_key(i, j)
        if self._rng.random() < self.eps:
            return self._rng.randrange(self.n_actions)
        qa = self.q.get(s, {})
        if not qa:
            if self.guidance_on and self._score_map is not None:
                ga = self._best_guided_action(i, j)
                if ga is not None:
                    return ga
            return self._rng.randrange(self.n_actions)
        best = max(qa, key=lambda a: qa[a])
        if self.guidance_on and self._score_map is not None:
            ga = self._best_guided_action(i, j)
            if ga is not None and qa.get(ga, 0.0) + 1.0 > qa.get(best, 0.0):
                return ga
        return best

    def _learn(self, s, a, r, ns):
        old = self.q.get(s, {}).get(a, 0.0)
        nq = self.q.get(ns, {})
        nq_max = max(nq.values()) if nq else 0.0
        self.q.setdefault(s, {})[a] = old + self.lr * (r + self.gamma * nq_max - old)

    # ---- 主循环 ----
    def run_episode(self, algo_seed: int = 42) -> dict:
        # 改进2：episode 级 RNG 隔离。所有随机性一律走 self._rng 实例，
        # 不再猴子补丁模块级 random（线程池下多线程共享模块函数会互相覆盖，
        # 导致同 seed 数字漂移）。self._rng 仍是 random.Random，调用顺序不变，
        # 故单线程结果与修复前逐点一致。
        self._rng = random.Random(algo_seed)
        return self._run()

    def _run(self):
        self.reset()
        self.oracle.reset_counter()
        o = self.oracle
        self.cur_cell = self._random_unvisited()

        while o.query_count < self.max_queries:
            if self.cur_cell is None:
                self.cur_cell = self._random_unvisited()
            i, j = self.cur_cell
            if not o.in_bounds(i, j):
                self.cur_cell = self._random_unvisited()
                continue
            self.visited_cells.add((i, j))

            pres, _ = o.query_cell(i, j)
            self._record_query(i, j, pres)
            if pres and self._distance_ok(i, j):
                self.found_cells.append((i, j))
                self._found_t.append(o.query_count)
                if self.mutation_detect:
                    self._last_hit_q = o.query_count
                self._last_hit_q_any = o.query_count
                self._last_found_cell = (i, j)
                self._stage3_idle = 0
                self._steps_since_jump = 0
                self._far_cooldown = False

            n = len(self.found_cells)
            if n >= self.target:
                break
            if self.stage == 1 and n >= self.stage2_min and self._coverage_ok():
                self.stage = 2
                # 优化项2：显式记录 Stage1 盲扫结束时的覆盖比例
                self.stage1_end_coverage = len(self.visited_cells) / max(o.nrows * o.ncols, 1)
            elif self.stage == 2 and n >= self.stage3_min:
                self.stage = 3
            if o.query_count >= self.max_queries:
                break

            # ---- 移动决策 ----
            if self.stage == 1:
                if self.stage1_prior:
                    nxt = self._stage1_prior_next(o)
                    self.cur_cell = nxt if nxt is not None else self._random_unvisited()
                else:
                    st = self.stage1_grid_step
                    if self.stage1_scan_mode == "serpentine":
                        nxt = (i + st, j) if i + st < o.nrows else (st, j + 1)
                        if nxt[1] >= o.ncols:
                            nxt = (nxt[0] % o.nrows, 0)
                        self.cur_cell = nxt
                    else:
                        # 改进1：spread 交错铺开（象限轮转）
                        self.cur_cell = self._stage1_spread_next(i, j, o)
            elif self.stage == 2:
                # 导向扫描：top-K 候选中随机（保留探索多样性，避免锁定单集群）
                k = self.cfg.get("stage2_topk", 8)
                cand = self._top_unvisited(k)
                # 五轮-M4：跟随模式激活时局部环候选优先（环空回退原 top-K）
                if self._follow_active():
                    fr = self._follow_ring(k)
                    if fr:
                        cand = fr
                # 改进3：运动外推——把漂移预测点注入候选首位（helper 内部已含全部门控）
                pc = self._motion_predicted_cell()
                if pc is not None and pc not in self.visited_cells and pc not in cand:
                    cand = [pc] + list(cand)
                if cand:
                    self.cur_cell = self._rng.choice(cand)
                else:
                    self.cur_cell = self._random_unvisited()
            else:
                # Stage3：远跳 / Q-learning 移动（导向叠加）
                self._stage3_idle += 1
                self._steps_since_jump += 1
                jumped = False
                # 迭代v2-C：跳落地后 post_jump_grace 步无发现即允许再次换区，
                # 避免在死区空跑一个 30 步间隔；远跳失败一次后退回正常间隔
                early_rejump = (self.post_jump_grace > 0 and self._has_jumped and not self._far_cooldown
                                and self._steps_since_jump >= self.post_jump_grace)
                # 五轮-M6：突变窗口内远跳间隔缩短（mutation_jump_interval）；OFF 时恒等于原值
                mut_active = (self.mutation_detect and self.mutation_disperse_jump
                              and self._mutation_streak() > self.mutation_streak_thresh)
                jump_interval = self.mutation_jump_interval if mut_active else self.stage3_jump_interval
                if ((self._stage3_idle >= jump_interval or early_rejump)
                        and self.found_cells):
                    if self._far_jump():
                        self._stage3_idle = 0
                        self._steps_since_jump = 0
                        self._has_jumped = True
                        self._far_cooldown = False
                        jumped = True
                    else:
                        self._far_cooldown = True
                if not jumped:
                    # 五轮-M4：跟随模式局部环直飞（环空立即回退原路径）
                    if self._follow_active():
                        fr = self._follow_ring(1)
                        if fr:
                            self.cur_cell = fr[0]
                            jumped = True
                if not jumped:
                    # 改进3：运动外推——无远跳时优先飞往漂移预测点（helper 内部已含全部门控；
                    # 落地后由下一轮主循环正常查询，不污染 Q-learning 记账）
                    pc = self._motion_predicted_cell()
                    if pc is not None and pc not in self.visited_cells:
                        self.cur_cell = pc
                        jumped = True
                if not jumped:
                    s = self._state_key(i, j)
                    a = self._choose_action(i, j)
                    di, dj = DIRS[a // len(STEPS)]
                    st = STEPS[a % len(STEPS)]
                    ni, nj = i + di * st, j + dj * st
                    if not o.in_bounds(ni, nj):
                        self.cur_cell = self._random_unvisited()
                        continue
                    self.cur_cell = (ni, nj)
                    self.visited_cells.add((ni, nj))
                    pres2, _ = o.query_cell(ni, nj)
                    self._record_query(ni, nj, pres2)
                    r2 = 10.0 if (pres2 and self._distance_ok(ni, nj)) else (
                        -5.0 if (ni, nj) in [(f[0], f[1]) for f in self.found_cells]
                        else self.curiosity_weight * 3.0)
                    if pres2 and self._distance_ok(ni, nj):
                        self.found_cells.append((ni, nj))
                        self._found_t.append(o.query_count)
                        if self.mutation_detect:
                            self._last_hit_q = o.query_count
                        self._last_hit_q_any = o.query_count
                        self._last_found_cell = (ni, nj)
                        self._stage3_idle = 0
                        self._steps_since_jump = 0
                        self._far_cooldown = False
                    ns = self._state_key(ni, nj)
                    self._learn(s, a, r2, ns)

            if o.query_count % 100 < 1 or len(self.found_cells) >= self.target:
                self.convergence.append((o.query_count, len(self.found_cells), self.stage))
            self.eps = max(self.eps_end, self.eps * self.eps_decay)

        self.convergence.append((o.query_count, len(self.found_cells), self.stage))
        return {
            "points_found": len(self.found_cells),
            "total_queries": o.query_count,
            "final_stage": self.stage,
            "found_cells": list(self.found_cells),
            "convergence": self.convergence,
            "feature_log": list(self._feature_log),
            # v3 覆盖指标显式化：访问网格总数 / 占研究区比例
            "covered_cells": len(self.visited_cells),
            "covered_ratio": len(self.visited_cells) / max(o.nrows * o.ncols, 1),
            # 优化项2：Stage1 盲扫结束时的覆盖比例（未结束为 None）
            "stage1_end_coverage": self.stage1_end_coverage,
        }


# ==================== 基线算法 ====================

def random_baseline(oracle: SpeciesOracle, config: dict, algo_seed: int) -> dict:
    rng = random.Random(algo_seed)
    target = int(config.get("target_points", 8))
    max_q = int(config.get("max_queries", 4000))
    min_d = float(config.get("min_dist_cells", 2.0))
    found = []
    oracle.reset_counter()
    conv = []
    while oracle.query_count < max_q and len(found) < target:
        i = rng.randrange(oracle.nrows)
        j = rng.randrange(oracle.ncols)
        pres, _ = oracle.query_cell(i, j)
        if pres:
            if all(math.hypot(i - fi, j - fj) >= min_d for fi, fj in found):
                found.append((i, j))
        if oracle.query_count % 100 < 1:
            conv.append((oracle.query_count, len(found)))
    conv.append((oracle.query_count, len(found)))
    return {"points_found": len(found), "total_queries": oracle.query_count,
            "final_stage": 1, "found_cells": found, "convergence": conv}


def grid_baseline(oracle: SpeciesOracle, config: dict, algo_seed: int) -> dict:
    target = int(config.get("target_points", 8))
    max_q = int(config.get("max_queries", 4000))
    min_d = float(config.get("min_dist_cells", 2.0))
    found = []
    oracle.reset_counter()
    conv = []
    step = int(config.get("grid_step", 2))
    i = 0
    while oracle.query_count < max_q and len(found) < target:
        j = (i // oracle.nrows) % oracle.ncols
        ii = i % oracle.nrows
        i += step
        pres, _ = oracle.query_cell(ii, j)
        if pres:
            if all(math.hypot(ii - fi, j - fj) >= min_d for fi, fj in found):
                found.append((ii, j))
        if oracle.query_count % 100 < 1:
            conv.append((oracle.query_count, len(found)))
    conv.append((oracle.query_count, len(found)))
    return {"points_found": len(found), "total_queries": oracle.query_count,
            "final_stage": 1, "found_cells": found, "convergence": conv}
