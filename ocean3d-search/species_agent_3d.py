# -*- coding: utf-8 -*-
"""species_agent_3d.py — 黑盒在线目标搜索框架的 3D 推广（26 方向 / 球壳远跳 / 26 邻域聚类）
对应 species_agent.py（2D）的维度推广：
    DIRS          8 方向   → 26 方向（3D 邻域）
    STEPS         4 步长   → 4 步长（保持）
    Stage1        2D 网格  → 3D 线性扫描（逐层推进）
    聚类          8 邻域   → 26 邻域（3D 分箱 BFS）
    环带候选      圆环     → 球壳
    远跳          8 方向   → 3D 球壳定向远跳
    Q 状态        (i//8,j//8) → (k//4,i//4,j//4) + 协变量分箱（加 z 层）
其余机制（分箱密度导向 / AUC 稀疏特征选择 / A11 自适应 / 分散约束 / 覆盖指标）保持不变。
"""
import math
import random

import numpy as np

from oracle3d import Oracle3D

# ---- 3D 动作空间：26 方向 × 4 步长 ----
DIRS = [(dk, di, dj) for dk in (-1, 0, 1) for di in (-1, 0, 1) for dj in (-1, 0, 1)
        if not (dk == 0 and di == 0 and dj == 0)]
STEPS = [2, 5, 10, 20]
N_BINS = 8


class Searcher3D:
    def __init__(self, oracle: Oracle3D, config: dict):
        self.oracle = oracle
        self.cfg = config
        self.target = int(config.get("target_points", 20))
        self.max_queries = int(config.get("max_queries", 10000))
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
        self.stage1_grid_step = int(config.get("stage1_grid_step", 3))
        # 改进1：冷启动盲扫铺开（与 2D 同口径）。"spread"=行带轮转交错（默认）；
        # "serpentine"=完整复现旧 3D 线性光栅。
        self.stage1_scan_mode = config.get("stage1_scan_mode", "spread")
        self.n_bins = int(config.get("n_bins", N_BINS))
        self._rng = random.Random(0)
        self.n_actions = len(DIRS) * len(STEPS)
        self.guidance_on = config.get("guidance_on", True)
        self.cluster_guidance = bool(config.get("cluster_guidance", False))
        # ---- 优化项：多簇分别追 + 覆盖预算 + 三级远跳（2D 同口径；False/0 复旧行为）----
        self.multicluster = bool(config.get("multicluster", True))
        self.cluster_ring_ratio = float(config.get("cluster_ring_ratio", 10.0))
        self.cluster_idle_limit = int(config.get("cluster_idle_limit", 12))
        self.blank_jump = bool(config.get("blank_jump", True))
        self.coverage_budget_ratio = float(config.get("coverage_budget_ratio", 0.12))
        # 迭代v2（与 2D 同口径）
        self.blank_far_ratio = float(config.get("blank_far_ratio", 0.3))
        self.post_jump_grace = int(config.get("post_jump_grace", 6))
        self.saturate_mask = bool(config.get("saturate_mask", False))
        self.mask_idle_limit = int(config.get("mask_idle_limit", 20))
        self.idle_blank_mix = int(config.get("idle_blank_mix", 8))
        self._cov_range = []
        for k in range(oracle.covs.shape[0]):
            v = oracle.covs[k]
            self._cov_range.append((float(np.nanmin(v)), float(np.nanmax(v))))
        self.reset()

    # ==================== 状态 ====================
    def reset(self):
        self.found_cells = []
        self.visited_cells = set()
        self.stage = 1
        self.q = {}
        self.eps = self.eps_start
        self._queries = []
        self._feature_log = []
        self._densities = None
        self._weights = {}
        self._score_map = None
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
        # 改进1：spread 盲扫游标（(k,i) 平面按交错列序访问，每列内扫 j；惰性初始化）
        self._s3d_inited = False
        self._s3d_plane0 = 0
        self._s3d_stride = 1
        self._s3d_offset = 0
        self._s3d_band = 0
        self._s3d_j = 0

    def _distance_ok(self, c):
        if not self.found_cells:
            return True
        for f in self.found_cells:
            if math.dist(c, f) < self.min_dist:
                return False
        return True

    # ==================== 密度导向（与 2D 一致） ====================
    def _auc(self, pos_vals, neg_vals):
        n, m = len(pos_vals), len(neg_vals)
        if n == 0 or m == 0:
            return 0.5
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
        while i < L:
            j = i
            while j + 1 < L and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        rsum = float(ranks[:n].sum())
        u = rsum - n * (n + 1) / 2.0
        auc = u / (n * m)
        return max(auc, 1.0 - auc)

    def _build_density(self):
        if len(self._queries) < self.stage2_min:
            return
        covs = np.array([q[0] for q in self._queries])
        pres = np.array([q[1] for q in self._queries], dtype=float)
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
            cnt = np.bincount(b, minlength=self.n_bins)
            pos = np.bincount(b, weights=pres, minlength=self.n_bins)
            nz = cnt > 0
            dens[nz] = pos[nz] / cnt[nz]
            dens[~nz] = pos.sum() / max(cnt.sum(), 1)
            self._densities[k] = (edges, dens)
            raw_w[k] = float(np.var(dens))

        if self.cfg.get("sparse_guidance", False):
            n_pos = int(pres.sum())
            n_q = len(self._queries)
            cold_start = int(self.cfg.get("cold_start_samples", 0))
            if n_pos >= self.cfg.get("sparse_min_pos", 5) and n_q >= cold_start:
                topk_base = int(self.cfg.get("sparse_topk", 6))
                n_vars = len(raw_w)
                topk = min(topk_base + max(0, (n_pos - 5) // 15), n_vars)
                ranked_v = sorted(raw_w.items(), key=lambda x: -x[1])[:12]
                pos_idx = np.where(pres > 0.5)[0]
                neg_idx = np.where(pres < 0.5)[0]
                auc_w = {}
                for k, _ in ranked_v:
                    auc_w[k] = self._auc(covs[pos_idx, k], covs[neg_idx, k])
                ranked_a = sorted(auc_w.items(), key=lambda x: -x[1])[:topk]
                keep = {k: max(w - 0.5, 0.0) ** 2 * 4.0 for k, w in ranked_a if w > 0.55}
                if keep:
                    raw_w = keep
                self._feature_log.append(
                    (len(self._queries), n_pos,
                     sorted(keep.items(), key=lambda x: -x[1])))

        tot = sum(raw_w.values())
        self._weights = {k: (w / tot if tot > 0 else 1.0 / len(raw_w))
                         for k, w in raw_w.items()}

    def _cell_score(self, c):
        if self._densities is None:
            return 0.0
        s = 0.0
        for k in self.cov_indices:
            edges, dens = self._densities[k]
            v = float(c[k])
            b = np.clip(np.digitize(v, edges) - 1, 0, self.n_bins - 1)
            s += self._weights.get(k, 0.0) * dens[b]
        return s

    def _build_score_map(self):
        o = self.oracle
        sm = np.zeros((o.K, o.L, o.J), dtype=np.float32)
        if self._densities is None:
            self._score_map = sm
            return
        for k in self.cov_indices:
            edges, dens = self._densities[k]
            b = np.clip(np.digitize(o.covs[k], edges) - 1, 0, self.n_bins - 1)
            sm += self._weights.get(k, 0.0) * dens[b]
        self._score_map = sm

    def _record_query(self, c, pres):
        self._queries.append((self.oracle.covs[:, c[0], c[1], c[2]].copy(), bool(pres)))
        n_q = len(self._queries)
        if self.stage >= 2 and (n_q % self.cfg.get("recalc_every", 20) == 0
                                or n_q == self.stage2_min):
            self._build_density()
            self._build_score_map()

    # ==================== 3D 聚类（26 邻域 BFS） ====================
    def _cluster_found_cells(self):
        o = self.oracle
        n = len(self.found_cells)
        if n < 3:
            return [list(self.found_cells)]
        bs_k = max(1, o.K // self.n_bins)
        bs_i = max(1, o.L // self.n_bins)
        bs_j = max(1, o.J // self.n_bins)
        bins = {}
        order = []
        for c in self.found_cells:
            b = (c[0] // bs_k, c[1] // bs_i, c[2] // bs_j)
            if b not in bins:
                bins[b] = []
                order.append(b)
            bins[b].append(c)
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
                bk, bi, bj = b
                for dk in (-1, 0, 1):
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            if dk == 0 and di == 0 and dj == 0:
                                continue
                            nb = (bk + dk, bi + di, bj + dj)
                            if nb in bins and nb not in seen:
                                seen.add(nb)
                                stack.append(nb)
            clusters.append(members)
        return clusters

    def _sphere_candidates(self, cluster, min_d, max_d):
        """以簇成员为中心、[min_d,max_d] 球壳内未访问格，按导向分降序去重"""
        o = self.oracle
        sm = self._score_map if self._score_map is not None else None
        cand = []
        r = int(max_d)
        for f in cluster:
            for dk in range(-r, r + 1):
                for di in range(-r, r + 1):
                    for dj in range(-r, r + 1):
                        d = math.sqrt(dk * dk + di * di + dj * dj)
                        if d < min_d or d > max_d:
                            continue
                        nk, ni, nj = f[0] + dk, f[1] + di, f[2] + dj
                        if not o.in_bounds(nk, ni, nj) or (nk, ni, nj) in self.visited_cells:
                            continue
                        s = float(sm[nk, ni, nj]) if sm is not None else 0.0
                        cand.append((s, nk, ni, nj))
        seen = set()
        uniq = []
        for s, nk, ni, nj in sorted(cand, key=lambda x: -x[0]):
            if (nk, ni, nj) in seen:
                continue
            seen.add((nk, ni, nj))
            uniq.append((nk, ni, nj))
        return uniq

    # ==================== 优化项1：多簇簿记（簇划分复用 26 邻域 BFS） ====================
    def _sync_clusters(self):
        """依据 found_cells 重建簇划分，沿用上一轮簇的生长时间/饱和状态。"""
        if not self.found_cells:
            self._clusters = []
            return
        raw = self._cluster_found_cells()
        prev = self._clusters
        out = []
        for members in raw:
            ck = sum(float(m[0]) for m in members) / len(members)
            ci = sum(float(m[1]) for m in members) / len(members)
            cj = sum(float(m[2]) for m in members) / len(members)
            cover = max(math.dist(m, (ck, ci, cj)) for m in members)
            best, bd = None, None
            for pc in prev:
                d = math.dist(pc["centroid"], (ck, ci, cj))
                if bd is None or d < bd:
                    best, bd = pc, d
            inherit = best is not None and bd <= cover + self.min_dist + 2
            last_q = best["last_grown_q"] if inherit else 0
            sat = best["saturated"] if inherit else False
            if self._last_found_cell in members:
                last_q = self.oracle.query_count
                sat = False
            out.append({"members": members, "centroid": (ck, ci, cj),
                        "cover_radius": cover, "last_grown_q": last_q,
                        "saturated": sat})
        for cl in out:
            if self.oracle.query_count - cl["last_grown_q"] > self.cluster_idle_limit:
                cl["saturated"] = True
        self._clusters = out

    def _ring_around_centroid(self, centroid, min_d, max_d):
        """以簇质心为中心、[min_d, max_d] 球壳内未访问格，按导向分降序。
        半径上界 = cluster_ring_ratio × min_dist（替换旧 max(5, min_dist×ratio)）。"""
        o = self.oracle
        sm = self._score_map
        ck, ci, cj = (int(round(centroid[0])), int(round(centroid[1])),
                      int(round(centroid[2])))
        r = int(max_d)
        cand = []
        for dk in range(-r, r + 1):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    d = math.sqrt(dk * dk + di * di + dj * dj)
                    if d < min_d or d > max_d:
                        continue
                    nk, ni, nj = ck + dk, ci + di, cj + dj
                    if not o.in_bounds(nk, ni, nj) or (nk, ni, nj) in self.visited_cells:
                        continue
                    s = float(sm[nk, ni, nj]) if sm is not None else 0.0
                    cand.append((s, nk, ni, nj))
        cand.sort(key=lambda x: -x[0])
        return [(nk, ni, nj) for _, nk, ni, nj in cand]

    def _effective_score_map(self):
        """迭代v2-B（与 2D 同口径）：idle > mask_idle_limit 的簇球域从导向分临时置零。"""
        if not self.multicluster or not self.saturate_mask or self._score_map is None:
            return self._score_map
        self._sync_clusters()
        stale = [c for c in self._clusters
                 if self.oracle.query_count - c["last_grown_q"] > self.mask_idle_limit]
        if not stale or len(stale) == len(self._clusters):
            return self._score_map
        sig = (id(self._score_map), self.mask_idle_limit,
               tuple(sorted((round(c["centroid"][0], 1), round(c["centroid"][1], 1),
                             round(c["centroid"][2], 1), round(c["cover_radius"], 1))
                            for c in stale)))
        if self._eff_sig == sig and self._eff_sm is not None:
            return self._eff_sm
        o = self.oracle
        sm = np.array(self._score_map, copy=True)
        for c in stale:
            ck, ci, cj = (int(round(c["centroid"][0])), int(round(c["centroid"][1])),
                          int(round(c["centroid"][2])))
            R = int(c["cover_radius"]) + 2
            for dk in range(-R, R + 1):
                for di in range(-R, R + 1):
                    for dj in range(-R, R + 1):
                        nk, ni, nj = ck + dk, ci + di, cj + dj
                        if o.in_bounds(nk, ni, nj):
                            sm[nk, ni, nj] = 0.0
        self._eff_sm = sm
        self._eff_sig = sig
        return sm

    def _blank_candidates(self, n=1):
        """迭代v2-A（3D 同口径）：粗网格步长 max(8, min(K,L,J)//8)，先过滤
        '距最近已发现点 ≥ far_dist' 的远距未访问格，再按全局导向分降序 top-8 随机 n。"""
        o = self.oracle
        step = max(8, min(o.K, o.L, o.J) // 8)
        far_dist = self.blank_far_ratio * min(o.K, o.L, o.J)
        sm = self._effective_score_map()
        far = []
        near = []
        for kk in range(0, o.K, step):
            for ii in range(0, o.L, step):
                for jj in range(0, o.J, step):
                    if (kk, ii, jj) in self.visited_cells:
                        continue
                    dmin = min(math.dist((kk, ii, jj), f) for f in self.found_cells) \
                        if self.found_cells else 1e9
                    s = float(sm[kk, ii, jj]) if sm is not None else 0.0
                    if dmin >= far_dist:
                        far.append((s, kk, ii, jj))
                    else:
                        near.append((dmin, kk, ii, jj))
        pool = far if far else near
        if not pool:
            return []
        if sm is not None and far:
            pool.sort(key=lambda x: -x[0])
        else:
            pool.sort(key=lambda x: -x[0])
        top = pool[:8]
        self._rng.shuffle(top)
        return [(kk, ii, jj) for _, kk, ii, jj in top[:n]]

    def _blank_jump(self):
        """三级远跳 L3：空白区开簇跳。跳距由域尺寸/簇间距决定，不受比例限制。"""
        picks = self._blank_candidates(1)
        if not picks:
            return False
        self.cur_cell = picks[0]
        return True

    def _cross_cluster_jump(self):
        """三级远跳 L2：跨簇转移跳——跳到另一未饱和簇质心球壳的随机候选。"""
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

    # ---- 改进1：冷启动盲扫铺开（spread，3D 适配：(k,i) 平面交错列序光栅）----
    def _stage1_spread_next_3d(self, c, o):
        """以起始 (kk0,ii0) 为锚，把 (k,i) 平面展平成 row_id=kk*L+ii，按交错步长
        访问行：先 row0, row0+stride, ..., 再 row0+1, ...；每行内 j 按 st 扫到底。
        小预算下首批查询即散布 (k,i) 平面，扫完一行才推进下一个交错行，不会原地卡死。
        纯算术确定性推进，不消耗 episode rng。"""
        st = self.stage1_grid_step
        nplane = o.K * o.L
        if not self._s3d_inited:
            kk0, ii0, jj0 = c
            B = int(self.cfg.get("stage1_spread_bands", max(4, nplane // 40)))
            B = max(2, min(B, nplane))
            self._s3d_plane0 = kk0 * o.L + ii0
            self._s3d_stride = max(1, nplane // B)
            self._s3d_offset = 0
            self._s3d_band = 0
            self._s3d_j = jj0 + st
            self._s3d_inited = True
        else:
            self._s3d_j += st
        if self._s3d_j >= o.J:
            self._s3d_j = st
            self._s3d_band += 1
            if self._s3d_band >= max(2, nplane // self._s3d_stride):
                self._s3d_band = 0
                self._s3d_offset += 1
        pidx = (self._s3d_plane0 + self._s3d_band * self._s3d_stride + self._s3d_offset) % nplane
        kk = pidx // o.L
        ii = pidx % o.L
        return (kk, ii, self._s3d_j)

    def _use_cluster_mode(self):
        if not self.cfg.get("adaptive", False):
            return self.cluster_guidance
        return len(self._cluster_found_cells()) >= 2

    def _top_unvisited(self, k: int):
        o = self.oracle
        sm = self._effective_score_map() if self._score_map is not None else np.zeros((o.K, o.L, o.J))
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
                        cands.extend(self._ring_around_centroid(cl["centroid"], min_d, max_d)[:per])
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
                # v2f 单簇：走旧全局导向 top-K；停滞 idle>=idle_blank_mix 才追加空白候选
                flat = np.argsort(sm.ravel())[::-1]
                out = []
                for idx in flat:
                    c = (int(idx // (o.L * o.J)), int((idx // o.J) % o.L), int(idx % o.J))
                    if c not in self.visited_cells:
                        out.append(c)
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
                    cands.extend(self._sphere_candidates(cl, min_d, max_d)[:per])
                if len(cands) < k:
                    flat = np.argsort(sm.ravel())[::-1]
                    for idx in flat:
                        c = (int(idx // (o.L * o.J)), int((idx // o.J) % o.L), int(idx % o.J))
                        if c not in self.visited_cells and c not in cands:
                            cands.append(c)
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
            c = (int(idx // (o.L * o.J)), int((idx // o.J) % o.L), int(idx % o.J))
            if c not in self.visited_cells:
                cands.append(c)
                if len(cands) >= k:
                    break
        return cands

    def _random_unvisited(self):
        o = self.oracle
        for _ in range(300):
            c = (self._rng.randrange(o.K), self._rng.randrange(o.L), self._rng.randrange(o.J))
            if c not in self.visited_cells:
                return c
        return (self._rng.randrange(o.K), self._rng.randrange(o.L), self._rng.randrange(o.J))

    # ==================== 3D 球壳定向远跳 ====================
    def _far_jump(self):
        if not self.found_cells:
            return False
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
                clusters = self._cluster_found_cells()
                for cl in clusters:
                    ring = self._sphere_candidates(cl, min_d, max_d)
                    for c in ring[:8]:
                        cands.append((float(sm[c[0], c[1], c[2]]), c))
                        if len(cands) >= 24:
                            break
                    if len(cands) >= 24:
                        break
            else:
                thr = float(np.percentile(sm, 80))
                hi_list = list(np.argwhere(sm.ravel() >= thr))
                self._rng.shuffle(hi_list)
                for idx in hi_list:
                    c = (int(idx[0] // (o.L * o.J)), int((idx[0] // o.J) % o.L), int(idx[0] % o.J))
                    if c in self.visited_cells:
                        continue
                    dmin = min(math.dist(c, f) for f in self.found_cells)
                    if min_d <= dmin <= max_d:
                        cands.append((float(sm[c[0], c[1], c[2]]), c))
                        if len(cands) >= 24:
                            break
        if not cands:
            # 回退：从已发现点向 26 方向做"方向内最优距离"定向扫描
            for f in self._rng.sample(self.found_cells, min(4, len(self.found_cells))):
                for dk, di, dj in DIRS:
                    best = None
                    for d in range(min_d, max_d + 1, max(1, (max_d - min_d) // 6)):
                        nk, ni, nj = f[0] + dk * d, f[1] + di * d, f[2] + dj * d
                        if o.in_bounds(nk, ni, nj) and (nk, ni, nj) not in self.visited_cells:
                            s = float(self._score_map[nk, ni, nj]) if self._score_map is not None else 0.0
                            if best is None or s > best[0]:
                                best = (s, nk, ni, nj)
                    if best:
                        cands.append((best[0], (best[1], best[2], best[3])))
        if not cands:
            return False
        cands.sort(key=lambda x: -x[0])
        top = cands[: min(6, len(cands))]
        _, c = self._rng.choice(top)
        self.cur_cell = c
        return True

    # ==================== Q-learning ====================
    def _state_key(self, c):
        if not self.cfg.get("cov_in_state", True):
            return (c[0] // 4, c[1] // 4, c[2] // 4)
        cc = self.oracle.covs[:, c[0], c[1], c[2]]
        bins = []
        for k in self.cov_indices:
            lo, hi = self._cov_range[k]
            if hi - lo < 1e-9:
                b = 0
            else:
                b = int((float(cc[k]) - lo) / (hi - lo) * 10)
                b = min(9, max(0, b))
            bins.append(b)
        return (c[0] // 4, c[1] // 4, c[2] // 4, tuple(bins))

    def _best_guided_action(self, c):
        sm = self._effective_score_map()
        best_a, best_s = None, -1.0
        for a in range(self.n_actions):
            dk, di, dj = DIRS[a // len(STEPS)]
            st = STEPS[a % len(STEPS)]
            nk, ni, nj = c[0] + dk * st, c[1] + di * st, c[2] + dj * st
            if not self.oracle.in_bounds(nk, ni, nj):
                continue
            s = float(sm[nk, ni, nj]) if sm is not None else 0.0
            if s > best_s:
                best_s, best_a = s, a
        return best_a

    def _choose_action(self, c):
        s = self._state_key(c)
        if self._rng.random() < self.eps:
            return self._rng.randrange(self.n_actions)
        qa = self.q.get(s, {})
        if not qa:
            if self.guidance_on and self._score_map is not None:
                ga = self._best_guided_action(c)
                if ga is not None:
                    return ga
            return self._rng.randrange(self.n_actions)
        best = max(qa, key=lambda a: qa[a])
        if self.guidance_on and self._score_map is not None:
            ga = self._best_guided_action(c)
            if ga is not None and qa.get(ga, 0.0) + 1.0 > qa.get(best, 0.0):
                return ga
        return best

    def _learn(self, s, a, r, ns):
        old = self.q.get(s, {}).get(a, 0.0)
        nq = self.q.get(ns, {})
        nq_max = max(nq.values()) if nq else 0.0
        self.q.setdefault(s, {})[a] = old + self.lr * (r + self.gamma * nq_max - old)

    # ==================== 主循环 ====================
    def run_episode(self, algo_seed: int = 42) -> dict:
        # 改进2：episode 级 RNG 隔离（与 2D 同口径）。所有随机性走 self._rng 实例，
        # 不再猴子补丁模块级 random。self._rng 仍是 random.Random，调用顺序不变。
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
            c = self.cur_cell
            if not o.in_bounds(*c):
                self.cur_cell = self._random_unvisited()
                continue
            self.visited_cells.add(c)
            pres, _ = o.query_cell(*c)
            self._record_query(c, pres)
            if pres and self._distance_ok(c):
                self.found_cells.append(c)
                self._last_found_cell = c
                self._stage3_idle = 0
                self._steps_since_jump = 0
                self._far_cooldown = False

            n = len(self.found_cells)
            if n >= self.target:
                break
            if self.stage == 1 and n >= self.stage2_min and self._coverage_ok():
                self.stage = 2
                # 优化项2：显式记录 Stage1 盲扫结束时的覆盖比例
                self.stage1_end_coverage = len(self.visited_cells) / max(o.total_cells, 1)
            elif self.stage == 2 and n >= self.stage3_min:
                self.stage = 3
            if o.query_count >= self.max_queries:
                break

            # ---- 移动决策 ----
            if self.stage == 1:
                st = self.stage1_grid_step
                if self.stage1_scan_mode == "serpentine":
                    # 3D 线性扫描：优先 (k,i) 行推进，j 循环；换行后 i 回绕；层内扫完推进 k
                    kk, ii, jj = c
                    jj2 = jj + st
                    if jj2 < o.J:
                        nxt = (kk, ii, jj2)
                    else:
                        ii2 = ii + st
                        if ii2 < o.L:
                            nxt = (kk, ii2, 0)
                        else:
                            nxt = (kk + 1, 0, 0) if kk + 1 < o.K else (0, 0, 0)
                    self.cur_cell = nxt
                else:
                    # 改进1：spread 行带轮转交错
                    self.cur_cell = self._stage1_spread_next_3d(c, o)
            elif self.stage == 2:
                k = self.cfg.get("stage2_topk", 8)
                cand = self._top_unvisited(k)
                self.cur_cell = self._rng.choice(cand) if cand else self._random_unvisited()
            else:
                self._stage3_idle += 1
                self._steps_since_jump += 1
                jumped = False
                # 迭代v2-C：跳落地后 post_jump_grace 步无发现即允许再次换区
                early_rejump = (self.post_jump_grace > 0 and self._has_jumped and not self._far_cooldown
                                and self._steps_since_jump >= self.post_jump_grace)
                if ((self._stage3_idle >= self.stage3_jump_interval or early_rejump)
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
                    s = self._state_key(c)
                    a = self._choose_action(c)
                    dk, di, dj = DIRS[a // len(STEPS)]
                    st = STEPS[a % len(STEPS)]
                    nk, ni, nj = c[0] + dk * st, c[1] + di * st, c[2] + dj * st
                    if not o.in_bounds(nk, ni, nj):
                        self.cur_cell = self._random_unvisited()
                        continue
                    nc = (nk, ni, nj)
                    self.cur_cell = nc
                    self.visited_cells.add(nc)
                    pres2, _ = o.query_cell(*nc)
                    self._record_query(nc, pres2)
                    r2 = 10.0 if (pres2 and self._distance_ok(nc)) else (
                        -5.0 if nc in self.found_cells else self.curiosity_weight * 3.0)
                    if pres2 and self._distance_ok(nc):
                        self.found_cells.append(nc)
                        self._last_found_cell = nc
                        self._stage3_idle = 0
                        self._steps_since_jump = 0
                        self._far_cooldown = False
                    ns = self._state_key(nc)
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
            "covered_cells": len(self.visited_cells),
            "covered_ratio": len(self.visited_cells) / max(o.total_cells, 1),
            # 优化项2：Stage1 盲扫结束时的覆盖比例（未结束为 None）
            "stage1_end_coverage": self.stage1_end_coverage,
        }


# ==================== 3D 基线算法 ====================

def random_baseline(oracle: Oracle3D, config: dict, algo_seed: int) -> dict:
    rng = random.Random(algo_seed)
    target = int(config.get("target_points", 20))
    max_q = int(config.get("max_queries", 10000))
    min_d = float(config.get("min_dist_cells", 1.0))
    found = []
    oracle.reset_counter()
    conv = []
    while oracle.query_count < max_q and len(found) < target:
        c = (rng.randrange(oracle.K), rng.randrange(oracle.L), rng.randrange(oracle.J))
        pres, _ = oracle.query_cell(*c)
        if pres:
            if all(math.dist(c, f) >= min_d for f in found):
                found.append(c)
        if oracle.query_count % 100 < 1:
            conv.append((oracle.query_count, len(found)))
    conv.append((oracle.query_count, len(found)))
    return {"points_found": len(found), "total_queries": oracle.query_count,
            "final_stage": 1, "found_cells": found, "convergence": conv}


def grid_baseline(oracle: Oracle3D, config: dict, algo_seed: int) -> dict:
    target = int(config.get("target_points", 20))
    max_q = int(config.get("max_queries", 10000))
    min_d = float(config.get("min_dist_cells", 1.0))
    found = []
    oracle.reset_counter()
    conv = []
    step = int(config.get("grid_step", 3))
    total = oracle.K * oracle.L * oracle.J
    idx = 0
    while oracle.query_count < max_q and len(found) < target:
        kk = (idx // (oracle.L * oracle.J)) % oracle.K
        ii = (idx // oracle.J) % oracle.L
        jj = idx % oracle.J
        idx = (idx + step) % total
        pres, _ = oracle.query_cell(kk, ii, jj)
        if pres:
            if all(math.dist((kk, ii, jj), f) >= min_d for f in found):
                found.append((kk, ii, jj))
        if oracle.query_count % 100 < 1:
            conv.append((oracle.query_count, len(found)))
    conv.append((oracle.query_count, len(found)))
    return {"points_found": len(found), "total_queries": oracle.query_count,
            "final_stage": 1, "found_cells": found, "convergence": conv}
