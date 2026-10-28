# -*- coding: utf-8 -*-
"""species_agent.py - black-box species search framework (species-version migration of the user framework)
Structural mapping from deep-sea-exploration/src/main_algorithm.py:
    cubiomes_wrapper  -> oracle.SpeciesOracle (black-box query: temperature -> covariates, deep-sea -> presence)
    database          -> presence + dispersion constraint inside the oracle
    state_manager     -> covariate binning + region (cov_in_state)
    Stage1/2/3        -> blind search -> parameter accumulation -> parameter-guided + Q-learning + long jump
    temperature_guidance -> binned-density guidance (uses queried-grid statistics of "parameter range -> presence rate")
    Stage3 long jump  -> when stuck, long jump from found points (jump distance = spacing threshold x ratio, variable-bound)
"""
import math
import random

import numpy as np

from oracle import SpeciesOracle

# ---- action space: 8 directions x 4 step sizes (grid units) ----
DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
STEPS = [2, 5, 10, 20]  # grid steps (~5km/13km/26km/52km)
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
        # improvement 1: cold-start blind sweep spread. "spread" = quadrant-rotated interleaved columns (default; under small budgets the spatial coverage
        # is uniform and independent of the starting column); "serpentine" = exact reproduction of the old serpentine raster (for A/B regression).
        self.stage1_scan_mode = config.get("stage1_scan_mode", "spread")
        self.n_bins = int(config.get("n_bins", N_BINS))   # bin count is configurable (ablation dimension)
        self._rng = random.Random(0)      # episode rng (injected by run_episode; guarantees reproducibility)
        self.n_actions = 8 * len(STEPS)
        self.guidance_on = config.get("guidance_on", True)
        self.cluster_guidance = bool(config.get("cluster_guidance", False))
        # ---- optimization items: per-cluster pursuit + coverage budget + three-level long jump ----
        # all on by default; passing False / 0 / 0.0 disables each item, reproducing old behavior (for A/B)
        self.multicluster = bool(config.get("multicluster", True))
        self.cluster_ring_ratio = float(config.get("cluster_ring_ratio", 10.0))
        self.cluster_idle_limit = int(config.get("cluster_idle_limit", 12))
        self.blank_jump = bool(config.get("blank_jump", True))
        self.coverage_budget_ratio = float(config.get("coverage_budget_ratio", 0.12))
        # iteration v2: guidance-blank jump distance threshold / post-jump grace steps / saturated-cluster mask switch
        self.blank_far_ratio = float(config.get("blank_far_ratio", 0.3))
        self.post_jump_grace = int(config.get("post_jump_grace", 6))
        self.saturate_mask = bool(config.get("saturate_mask", False))
        # iteration v2.1: delayed mask application (remove from the guidance map only after idle exceeds this value, leaving slow-start clusters exploration time)
        self.mask_idle_limit = int(config.get("mask_idle_limit", 20))
        self.idle_blank_mix = int(config.get("idle_blank_mix", 8))
        # improvement 3: motion extrapolation (default off). when on, estimates a drift velocity vector from the timestamps of found points,
        # injecting the predicted position into Stage 2/3 candidates for moving targets such as maritime search-and-rescue. off = bitwise identical to variant F.
        self.motion_extrapolation = bool(config.get("motion_extrapolation", False))
        # time decay / forgetting mechanism (round 4): historical samples fade out over time; recent observations dominate the guidance map.
        # time_decay=False is pointwise equivalent to the round-2 version.
        self.time_decay = bool(config.get("time_decay", False))
        self.decay_mode = config.get("decay_mode", "exp")          # "exp" / "window"
        self.decay_tau = float(config.get("decay_tau", 300.0))     # exponential time constant (in queries)
        self.decay_window = int(config.get("decay_window", 400))    # sliding-window size
        self.decay_min_weight = float(config.get("decay_min_weight", 0.05))  # AUC time-filter threshold
        self.q_decay_tau = float(config.get("q_decay_tau", 200.0)) # Q-table decay time constant
        self.q_decay_every = int(config.get("q_decay_every", 25))  # Q-table decay period (in queries)
        # round-4 M1: mutation detection + empty-window compression (all False by default; off = pointwise equivalent to the round-3 version)
        self.mutation_detect = bool(config.get("mutation_detect", False))
        self.mutation_streak_thresh = int(config.get("mutation_streak_thresh", 40))
        self.mutation_accel = float(config.get("mutation_accel", 2.0))
        self.mutation_ramp = int(config.get("mutation_ramp", 40))
        self.mutation_auto_motion = bool(config.get("mutation_auto_motion", False))
        # round-4 M2: Stage-1 prior focusing (default False)
        self.stage1_prior = bool(config.get("stage1_prior", False))
        self.stage1_prior_cov = int(config.get("stage1_prior_cov", 8))
        # round-4 M3: high-dimensional weak-signal weighting (default False)
        self.weak_signal_boost = bool(config.get("weak_signal_boost", False))
        self.weak_signal_min_auc = float(config.get("weak_signal_min_auc", 0.52))
        # round-5 M4: follow mode (default False; off is bitwise equivalent to round 4)
        self.follow_mode = bool(config.get("follow_mode", False))
        self.follow_window = int(config.get("follow_window", 24))
        self.follow_radius = float(config.get("follow_radius", 8.0))
        # round-5 M6: mutation-drought scatter jump (default False)
        self.mutation_disperse_jump = bool(config.get("mutation_disperse_jump", False))
        self.mutation_jump_interval = int(config.get("mutation_jump_interval", 15))
        # per-covariate full-grid min/max (for state binning; adaptive)
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
        # data accumulation: (covariate vector, presence) samples of queried cells
        self._queries = []          # [(cov_vec, presence_bool)]
        self._feature_log = []      # feature-selection diagnostics: (n_q, n_pos, [(cov_idx, weight)...])
        self._densities = None      # {k: (edges, density)}
        self._weights = {}          # {k: discriminative weight}
        self._score_map = None      # full-grid guidance score
        self._stage3_idle = 0
        self.cur_cell = None
        self.convergence = []
        # multi-cluster bookkeeping (optimization 1): [{members, centroid, cover_radius, last_grown_q, saturated}]
        self._clusters = []
        self._last_found_cell = None      # new found cell this round (for cluster-growth accounting)
        self.stage1_end_coverage = None   # coverage ratio at the end of Stage 1 (explicitly recorded by optimization 2)
        # iteration v2: post-jump rhythm / saturated-cluster mask cache
        self._steps_since_jump = 0
        self._has_jumped = False
        self._far_cooldown = False
        self._eff_sm = None
        self._eff_sig = None
        # improvement 1: spread blind-sweep cursor (interleaved column raster; lazily initialized when Stage 1 starts)
        self._s1_j0 = None
        self._s1_band_w = 0
        self._s1_B = 0
        self._s1_offset = 0
        self._s1_band = 0
        self._s1_row = 0
        # improvement 3: timestamps of found points (= query_count at discovery), kept in parallel with found_cells
        self._found_t = []
        # round 4: timestamps parallel to _queries (sample count at record time); only read by the time_decay branch
        self._query_t = []
        # round-4 M1: mutation-detection bookkeeping (query_count of the last hit)
        self._last_hit_q = -1
        # round-5 M4: follow-mode bookkeeping (query_count of the last hit, without mutation_detect gating)
        self._last_hit_q_any = -1
        # round-4 M2: Stage-1 prior-focus bookkeeping (lazily built)
        self._s1_prior_order = None
        self._s1_prior_pos = 0

    # ---- dispersion constraint ----
    def _distance_ok(self, i, j):
        if not self.found_cells:
            return True
        for fi, fj in self.found_cells:
            if math.hypot(i - fi, j - fj) < self.min_dist:
                return False
        return True

    # ---- binned-density guidance (uses all queried positive/negative samples; weighted by variable discriminativeness) ----
    # ---- discriminative screening: Mann-Whitney AUC (noise variables ~0.5, immune to small-sample fluctuation) ----
    def _auc(self, pos_vals, neg_vals):
        n, m = len(pos_vals), len(neg_vals)
        if n == 0 or m == 0:
            return 0.5
        # sampling controls the compute cost (uses the episode rng; reproducible)
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
        while i < L:                       # average rank for ties
            j = i
            while j + 1 < L and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        rsum = float(ranks[:n].sum())
        u = rsum - n * (n + 1) / 2.0
        auc = u / (n * m)
        return max(auc, 1.0 - auc)        # direction-agnostic; keep only the separability

    # ---- round 4: time decay / forgetting mechanism ----
    def _decay_weights(self, accel=True):
        """Return a float weight vector of length n (age of sample t = n-1 - _query_t[t]).
        time_decay=False returns all-ones; window mode zeroes weights beyond the window; exp mode decays exponentially.
        accel=False skips the M1 mutation acceleration (used to decouple the AUC parameter / n_pos gate)."""
        n = len(self._queries)
        if not self.time_decay or n == 0:
            return np.ones(n, dtype=np.float64)
        ages = (n - 1) - np.asarray(self._query_t, dtype=np.float64)
        # round-4 M1: enlarge the effective age of old samples in a mutation drought (at most x(1+mutation_accel)) to fade them out faster
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

    # ---- round-4 M1: mutation detection helpers ----
    def _mutation_streak(self):
        """Current "global zero-hit" query count (in the sample-index domain: n-1 - last_hit_q)."""
        if not self.mutation_detect:
            return 0
        n = len(self._queries)
        if self._last_hit_q < 0:
            return n - 1 if n > 0 else 0
        return max((n - 1) - self._last_hit_q, 0)

    # ---- round-5 M4: follow-mode helpers ----
    def _follow_active(self):
        """Follow mode active: follow_mode=True and the last hit is within follow_window queries.
        Short-circuits to False when OFF (reads no new bookkeeping; bitwise equivalent)."""
        if not self.follow_mode or self._last_hit_q_any < 0:
            return False
        return (self.oracle.query_count - self._last_hit_q_any) < self.follow_window

    def _follow_ring(self, k):
        """Anchor the ring band [min_dist, follow_radius] around the latest found point, unvisited cells in random order.
        Empty ring returns [] (the caller immediately falls back to the original path). Only called when follow_mode is ON."""
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

    # ---- round-5 M6: mutation-drought scatter jump ----
    def _mutation_disperse_jump(self):
        """Inside a mutation window: uniformly pick one unvisited coarse-grid cell at distance >= far_dist from all found points.
        Deliberately ignores the guidance score (after a mutation the guidance map still points to the old region, so a guided jump would be pulled back)."""
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
        """Call the original _auc after filtering by weight >= decay_min_weight.
        Returns 0.5 if either side is empty after filtering."""
        keep_p = w_pos >= self.decay_min_weight
        keep_n = w_neg >= self.decay_min_weight
        pv = pos_vals[keep_p]
        nv = neg_vals[keep_n]
        if len(pv) == 0 or len(nv) == 0:
            return 0.5
        return self._auc(pv, nv)

    def _decay_q(self):
        """Multiply all q[s][a] by an exponential decay factor. Pure arithmetic; consumes no self._rng.
        q_decay_tau>=1e7 is treated as infinity (no decay; returns directly for numerical equivalence)."""
        if self.q_decay_tau <= 0 or self.q_decay_tau >= 1e7:
            return
        factor = math.exp(-self.q_decay_every / max(self.q_decay_tau, 1e-9))
        for s in self.q:
            for a in self.q[s]:
                self.q[s][a] *= factor

    def _build_density(self):
        """Bin each covariate into N_BINS bins; density = presence queries in bin / queries in bin;
        also computes a separability weight per variable; highly discriminative variables get larger weights.
        v2 sparsification (sparse_guidance=True):
          once enough positives exist, first coarse-filter top-12 by bin-density variance, then fine-filter by AUC,
          keeping only statistically significant variables (AUC>0.55), with weights compressed by the power law (AUC-0.5)^2.
          topK relaxes progressively with more positives: more samples -> more reliable estimates -> more retained variables.
          noise variables with AUC~0.5 are naturally excluded -> more parameters only enlarge the candidate pool, and the online selector picks the useful subset."""
        if len(self._queries) < self.stage2_min:
            return
        covs = np.array([q[0] for q in self._queries])       # (n, K)
        pres = np.array([q[1] for q in self._queries], dtype=float)
        # round 4: time-decay weight vector (all-ones when time_decay=False)
        w = self._decay_weights()                  # density guidance: accelerated in mutation periods
        w_sel = self._decay_weights(accel=False)   # round-4 M1 decoupled: non-accelerated weights for the AUC parameter / n_pos gate
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
            # smoothing: bins without queries use the global presence rate
            dens[~nz] = pos.sum() / max(cnt.sum(), 1)
            self._densities[k] = (edges, dens)
            raw_w[k] = float(np.var(dens))   # coarse filter: bin-density variance

        # ---- v2 sparsification: AUC fine-filter + power-law compression + progressive topK ----
        if self.cfg.get("sparse_guidance", False):
            if self.time_decay:
                n_pos = int(round(float(np.sum(pres * w_sel))))
            else:
                n_pos = int(pres.sum())
            n_q = len(self._queries)
            # v3 cold start (default off; keeps v2 reproducible): when total samples are insufficient, use full weighting
            # (AUC is noisy under small samples), enabling sparsification only above the threshold; takes effect only when the experiment script explicitly
            # passes cold_start_samples>0
            cold_start = int(self.cfg.get("cold_start_samples", 0))
            if n_pos >= self.cfg.get("sparse_min_pos", 5) and n_q >= cold_start:
                topk_base = int(self.cfg.get("sparse_topk", 6))
                n_vars = len(raw_w)
                # more positives -> more reliable discriminative estimates -> more valid variables can be kept
                topk = min(topk_base + max(0, (n_pos - 5) // 15), n_vars)
                # 1) variance coarse-filter top-12 (fast), avoiding full AUC over 19 variables
                ranked_v = sorted(raw_w.items(), key=lambda x: -x[1])[:12]
                # 2) AUC fine-filter: noise variables ~0.5; true signals stably >0.55
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
                # 3) keep only significant variables; weight = (AUC-0.5)^2 * 4 (soft sparsification: weak signals near zero)
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
                # conservative fallback: if no significant variable (extreme samples), keep top1 so guidance never fully dies
                # feature-selection diagnostics: record this round's representative features and weights (descending by weight)
                self._feature_log.append(
                    (len(self._queries), n_pos,
                     sorted(keep.items(), key=lambda x: -x[1])))

        tot = sum(raw_w.values())
        self._weights = {k: (w / tot if tot > 0 else 1.0 / len(raw_w))
                         for k, w in raw_w.items()}

    def _cell_score(self, i, j):
        """Grid guidance score: presence density of the covariate bin, weighted by variable discriminativeness"""
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
        self._query_t.append(len(self._queries) - 1)  # timestamp = sample index (distance from the newest sample is the age)
        # density table / guidance map rebuilt every recalc_every queries (avoid O(KxN) full recomputation slowing 19 parameters)
        n_q = len(self._queries)
        if self.stage >= 2 and (n_q % self.cfg.get("recalc_every", 20) == 0
                                or n_q == self.stage2_min):
            self._build_density()
            self._build_score_map()
        # round 4: periodic Q-table decay (placed after the density-rebuild block)
        if (self.time_decay and self.q_decay_tau > 0
                and n_q % self.q_decay_every == 0):
            self._decay_q()

    # ---- cluster-aware guidance (multimodal: candidates spread over clusters to avoid the global guidance locking onto one cluster) ----
    def _cluster_found_cells(self):
        """Bin-grid connected clustering (8-neighborhood BFS): partition found points into spatial clusters"""
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
        """Cluster hot zone: unvisited cells in the [min_d, max_d] ring band around cluster members, deduplicated in descending global guidance score"""
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

    # ---- optimization 1: multi-cluster bookkeeping (reuses _cluster_found_cells bin-grid 8-neighborhood BFS) ----
    def _sync_clusters(self):
        """Rebuild the cluster partition from found_cells, carrying over each cluster's growth time and saturation state.
        Each cluster dict: {members, centroid, cover_radius, last_grown_q, saturated}.
        New points fall into clusters naturally via bin-grid BFS; the cluster containing this round's new point refreshes last_grown_q and
        unsaturates; consecutive queries without a new point beyond cluster_idle_limit mark it saturated."""
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
        """Unvisited cells in the [min_d, max_d] ring band around the cluster centroid, deduplicated in descending global guidance score.
        Upper radius = cluster_ring_ratio x min_dist (replacing the old near-sighted max(5, min_dist x jump_max_ratio));
        single-center scanning replaces the per-member loop, saving compute."""
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
        """Iteration v2-B: temporarily zero the guidance score of the cover regions of clusters idle too long (idle > mask_idle_limit),
        preventing Q/guidance from pulling the search back into exhausted clusters; gives slow-start clusters a mask_idle_limit-step exploration window.
        Returns score_map unchanged when there is no cluster to mask / all clusters to mask / multicluster=False."""
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
        """Iteration v2-A: guidance-blank jump candidates. Coarse-grid step max(8, min(nrows,ncols)//8);
        first filter for distant unvisited cells 'at distance >= far_dist (=blank_far_ratio x min(nrows,ncols)) from the nearest found point',
        then take the top-8 by global guidance score and pick n at random - jumping toward
        'far regions that look like target habitat' rather than random dead zones. Falls back to farthest sampling when the guidance map is absent."""
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
            # distant candidates in descending guidance score
            pool.sort(key=lambda x: -x[0])
            top = pool[:8]
        else:
            # no guidance map or no distant candidates: fall back to farthest sampling
            pool.sort(key=lambda x: -x[0])
            top = pool[:8]
        self._rng.shuffle(top)
        return [(i, j) for _, i, j in top[:n]]

    def _blank_jump(self):
        """Three-level long jump L3: open-cluster jump in blank regions. Jump distance is set by domain size / cluster spacing and is not
        bounded by the [jump_min_ratio, jump_max_ratio] ratio."""
        picks = self._blank_candidates(1)
        if not picks:
            return False
        self.cur_cell = picks[0]
        return True

    def _cross_cluster_jump(self):
        """Three-level long jump L2: cross-cluster transfer jump - jump to a random candidate on the outer ring of another unsaturated cluster."""
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
        """Optimization 2: Stage-1 blind-sweep budget floor. coverage_budget_ratio<=0 disables it (reproduces old behavior)"""
        r = self.coverage_budget_ratio
        if r <= 0:
            return True
        return self.oracle.query_count >= int(self.max_queries * r)

    # ---- improvement 1: cold-start blind-sweep spread (interleaved-column raster) ----
    def _stage1_spread_next(self, i, j, o):
        """Anchor at a random starting column j and visit columns in "interleaved order": first j0, j0+band_w, ...,
        j0+(B-1)*band_w (B uniformly spread columns), then fill in the offsets j0+1, j0+1+band_w, ... .
        Within each column, rows are swept downward at stage1_grid_step; only after one column finishes does the sweep move to the next interleaved column.
        Under small budgets the first batch of queries already spreads across the column axis (independent of the starting column), without the stall
        of a fixed-anchor column re-scanning in place. Pure arithmetic; consumes no episode rng."""
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

    # ---- round-4 M2: Stage-1 prior focusing ----
    def _stage1_prior_next(self, o):
        """Return the next unvisited cell in descending covs[stage1_prior_cov]; returns None when all are visited.
        Deterministic (no RNG); builds the sorted order once on first call."""
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

    # ---- improvement 3: motion extrapolation ----
    def _motion_predicted_cell(self):
        """Estimate the drift velocity vector from the timestamps of found points (median of adjacent displacement/time deltas),
        extrapolating the target's current position. Under a static oracle the displacement ~0 -> the prediction ~ the last found point (already visited,
        so it will not be re-injected), making the module naturally inert. Returns (pi,pj) or None."""
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

    # ---- adaptive switch (A11): online diagnosis of the spatial structure of found points ----
    # multi-cluster (>=2 clusters) -> cluster-spread guidance (prevents the global guidance from locking onto one high-score side);
    # single cluster (1 cluster) -> global guidance (covariates drive monotonically; guidance is effective).
    # with adaptive=True no manual cluster_guidance is needed; the machine switches automatically.
    def _use_cluster_mode(self):
        if not self.cfg.get("adaptive", False):
            return self.cluster_guidance
        clusters = self._cluster_found_cells()
        return len(clusters) >= 2

    # ---- candidate grid ----
    def _top_unvisited(self, k: int):
        o = self.oracle
        sm = self._effective_score_map() if self._score_map is not None else np.zeros((o.nrows, o.ncols))
        # ---- optimization 1: per-cluster pursuit (multicluster=True takes priority; otherwise the old logic runs entirely) ----
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
                # fallback: all clusters saturated -> drop to the old global guidance below
            else:
                # v2f single cluster: old global guidance top-K (CHL-type single dense hot-spot cluster);
                # only when that cluster stalls with idle>=idle_blank_mix do we add blank candidates to scatter and open a new cluster
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

    # ---- Stage3 long jump (variable-bound jump distance: threshold x ratio; directed to uncovered high-score bands to improve cross-cluster coverage) ----
    # v3 directed long jump: among the top-20% unvisited cells by global guidance score, find "cells whose distance to the nearest found point
    # falls in the jump range" as high-score target bands (cross-cluster direct reach); when they are insufficient, fall back to an 8-direction
    # directed scan + score ranking. Randomness is kept (random within top-N) to avoid locking onto one high-score cluster.
    def _far_jump(self):
        if not self.found_cells:
            return False
        # round-5 M6: scatter jump first in a mutation window (disperse to new regions, bypassing the old guidance bias)
        if (self.mutation_detect and self.mutation_disperse_jump
                and self._mutation_streak() > self.mutation_streak_thresh):
            if self._mutation_disperse_jump():
                return True
        # ---- optimization 3: three-level long jump (L2 cross-cluster transfer / L3 blank-cluster opening first; falls back to the normal long jump) ----
        if self.multicluster:
            self._sync_clusters()
            has_sat = any(cl["saturated"] for cl in self._clusters)
            # iteration v2-C: no discovery within post_jump_grace steps after landing = landed in a dead zone; trigger another region switch
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
                # cluster-aware long jump: uniform candidates per cluster hot zone (cross-cluster direct reach; avoids locking one high-score side)
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
                # global high-score unvisited cells (candidate target bands)
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
            # fallback: 8-direction "best-distance-within-direction" directed scan from found points
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
        # high-guidance candidates first (with randomness to avoid locking)
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

    # ---- main loop ----
    def run_episode(self, algo_seed: int = 42) -> dict:
        # improvement 2: episode-level RNG isolation. All randomness goes through the self._rng instance,
        # no longer monkey-patching the module-level random (under a thread pool, sharing module functions across threads overwrites each other,
        # causing same-seed number drift). self._rng is still a random.Random with unchanged call order,
        # so single-threaded results are pointwise identical to the pre-fix version.
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
                # optimization 2: explicitly record the coverage ratio when the Stage-1 blind sweep ends
                self.stage1_end_coverage = len(self.visited_cells) / max(o.nrows * o.ncols, 1)
            elif self.stage == 2 and n >= self.stage3_min:
                self.stage = 3
            if o.query_count >= self.max_queries:
                break

            # ---- movement decision ----
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
                        # improvement 1: spread interleaved opening (quadrant rotation)
                        self.cur_cell = self._stage1_spread_next(i, j, o)
            elif self.stage == 2:
                # guided scan: random among top-K candidates (keeps exploration diversity; avoids locking onto one cluster)
                k = self.cfg.get("stage2_topk", 8)
                cand = self._top_unvisited(k)
                # round-5 M4: when follow mode is active, local-ring candidates take priority (empty ring falls back to the original top-K)
                if self._follow_active():
                    fr = self._follow_ring(k)
                    if fr:
                        cand = fr
                # improvement 3: motion extrapolation - inject the drift-predicted point at the head of the candidates (all gating inside the helper)
                pc = self._motion_predicted_cell()
                if pc is not None and pc not in self.visited_cells and pc not in cand:
                    cand = [pc] + list(cand)
                if cand:
                    self.cur_cell = self._rng.choice(cand)
                else:
                    self.cur_cell = self._random_unvisited()
            else:
                # Stage3: long jump / Q-learning movement (guidance stacked)
                self._stage3_idle += 1
                self._steps_since_jump += 1
                jumped = False
                # iteration v2-C: allow another region switch when no discovery within post_jump_grace steps after landing,
                # avoiding dead-zone idle for a full 30-step interval; a failed long jump falls back to the normal interval once
                early_rejump = (self.post_jump_grace > 0 and self._has_jumped and not self._far_cooldown
                                and self._steps_since_jump >= self.post_jump_grace)
                # round-5 M6: long-jump interval shrinks inside a mutation window (mutation_jump_interval); OFF keeps the original value always
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
                    # round-5 M4: follow-mode local-ring direct flight (empty ring immediately falls back to the original path)
                    if self._follow_active():
                        fr = self._follow_ring(1)
                        if fr:
                            self.cur_cell = fr[0]
                            jumped = True
                if not jumped:
                    # improvement 3: motion extrapolation - when no long jump, fly to the drift-predicted point first (all gating inside the helper;
                    # after landing, the next main-loop round queries normally; no Q-learning bookkeeping is polluted)
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
            # v3 coverage metric made explicit: visited cells / share of the study region
            "covered_cells": len(self.visited_cells),
            "covered_ratio": len(self.visited_cells) / max(o.nrows * o.ncols, 1),
            # optimization 2: coverage ratio at the end of the Stage-1 blind sweep (None if not finished)
            "stage1_end_coverage": self.stage1_end_coverage,
        }


# ==================== Baseline algorithms ====================

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
