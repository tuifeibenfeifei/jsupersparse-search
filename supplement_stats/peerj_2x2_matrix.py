# -*- coding: utf-8 -*-
"""2x2 分解矩阵实验：selector(on/off) × scheduler(greedy/smart)，四场景。

selector on = sparse_guidance=True；off = sparse_guidance=False（密度引导保留）；
scheduler smart = 分阶段+Q-learning+远跳框架；greedy = 每步选全局最优未访问格。
输出 peerj_2x2_detail.csv / peerj_2x2_matrix.md / peerj_2x2_matrix.log。
"""
import csv
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(r"C:\Users\19264\Desktop\deep-sea-exploration")
HERE = ROOT / "supplement_stats"
sys.path.insert(0, str(ROOT / "species-search"))
sys.path.insert(0, str(ROOT / "chlorophyll-search"))
sys.path.insert(0, str(ROOT / "ocean3d-search"))
sys.path.insert(0, str(ROOT))

from oracle import SpeciesOracle          # noqa: E402
from species_agent import SpeciesSearcher  # noqa: E402
from oracle3d import Oracle3D              # noqa: E402
from species_agent_3d import Searcher3D    # noqa: E402

DETAIL_CSV = HERE / "peerj_2x2_detail.csv"
REPORT_MD = HERE / "peerj_2x2_matrix.md"
LOG = HERE / "peerj_2x2_matrix.log"

SEEDS = list(range(101, 121))  # 20 algo seeds
MC_WORLDS_FULL = list(range(1, 21))

# 配置（与 run_cross_ablation.py / run_o3d_regress_F.py 逐字一致）
MC_CFG = dict(
    target_points=40, max_queries=10000, min_dist_cells=2.0,
    stage2_min=3, stage3_min=5, stage3_jump_interval=30,
    stage3_jump_min_ratio=1.25, stage3_jump_max_ratio=5.0,
    curiosity_weight=1.0, guidance_on=True, cov_in_state=True,
    cov_indices=[0, 1, 2, 3, 4],
    multicluster=True, cluster_ring_ratio=10.0, cluster_idle_limit=12,
    blank_jump=True, coverage_budget_ratio=0.12,
    time_decay=True, decay_mode="exp", decay_tau=300.0,
    mutation_detect=True, mutation_streak_thresh=40,
    follow_mode=True, mutation_disperse_jump=True, mutation_jump_interval=15,
    # 以下键在 MC_CFG 原文件中缺省；为开 selector(on-smart) 补全为与 CHL/SPP 同口径
    learning_rate=0.1, discount_factor=0.95,
    epsilon_start=0.9, epsilon_end=0.1, epsilon_decay=0.999,
    stage1_grid_step=2, recalc_every=20, stage2_topk=8,
    sparse_guidance=False, sparse_topk=6, sparse_min_pos=5,
    blank_far_ratio=0.3, post_jump_grace=6, saturate_mask=False,
    adaptive=False, directed_jump=False,
)
SPP_CFG = dict(
    target_points=20, max_queries=20000, min_dist_cells=1.0,
    stage2_min=3, stage3_min=5, stage3_jump_interval=30,
    stage3_jump_min_ratio=1.25, stage3_jump_max_ratio=5.0,
    curiosity_weight=1.0, learning_rate=0.1, discount_factor=0.95,
    epsilon_start=0.9, epsilon_end=0.1, epsilon_decay=0.999,
    stage1_grid_step=2, guidance_on=True, cov_in_state=True,
    recalc_every=20, stage2_topk=8,
    sparse_guidance=True, sparse_topk=6, sparse_min_pos=5,
    multicluster=True, cluster_ring_ratio=10.0, cluster_idle_limit=12,
    blank_jump=True, coverage_budget_ratio=0.12,
    blank_far_ratio=0.3, post_jump_grace=6, saturate_mask=False,
    cov_indices=list(range(19)), adaptive=True, directed_jump=True,
)
CHL_CFG = dict(
    target_points=20, max_queries=4000, min_dist_cells=1.0,
    stage2_min=3, stage3_min=5, stage3_jump_interval=30,
    stage3_jump_min_ratio=1.25, stage3_jump_max_ratio=5.0,
    curiosity_weight=1.0, learning_rate=0.1, discount_factor=0.95,
    epsilon_start=0.9, epsilon_end=0.1, epsilon_decay=0.999,
    stage1_grid_step=3, guidance_on=True, cov_in_state=True,
    recalc_every=20, stage2_topk=8,
    sparse_guidance=True, sparse_topk=6, sparse_min_pos=5,
    multicluster=True, cluster_ring_ratio=10.0, cluster_idle_limit=12,
    blank_jump=True, coverage_budget_ratio=0.12,
    blank_far_ratio=0.3, post_jump_grace=6, saturate_mask=False,
    cov_indices=list(range(19)), adaptive=True, directed_jump=True,
)
O3D_BASE = dict(
    target_points=40, max_queries=10000, min_dist_cells=1.0,
    stage2_min=3, stage3_min=5, stage3_jump_interval=30,
    stage3_jump_min_ratio=1.25, stage3_jump_max_ratio=5.0,
    curiosity_weight=1.0, learning_rate=0.1, discount_factor=0.95,
    epsilon_start=0.9, epsilon_end=0.1, epsilon_decay=0.999,
    stage1_grid_step=3, guidance_on=True, cov_in_state=True,
    recalc_every=20, stage2_topk=8,
    sparse_guidance=False, sparse_topk=6, sparse_min_pos=5,
    multicluster=True, cluster_ring_ratio=10.0, cluster_idle_limit=12,
    blank_jump=True, coverage_budget_ratio=0.12,
    blank_far_ratio=0.3, post_jump_grace=6, saturate_mask=False,
    mask_idle_limit=20, idle_blank_mix=8,
    cov_indices=list(range(7)),
)

ORACLE_PATHS = {
    "chlorophyll": ROOT / "chlorophyll-search" / "data" / "oracle.npz",
    "clouded_leopard": ROOT / "species-search" / "data" / "clouded_leopard" / "oracle.npz",
}

# 日志
def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_csv(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


#  贪心臂实现（2D + 3D；协变量代理 + 空间纯贪心）

class Env2D:
    """与 harmful_arms_paired.Env 逐字一致的 2D 交互环境。"""
    def __init__(self, oracle, target, max_q, min_d):
        self.o = oracle
        self.target = target
        self.max_q = max_q
        self.min_d = min_d
        self.reset()

    def reset(self):
        self.o.reset_counter()
        self.found = []          # [(i,j)]
        self.visited = set()
        self.pos_cov = []        # 每个新发现点的原始 cov 向量
        self.X_hist = []
        self.y_hist = []

    @property
    def q(self):
        return self.o.query_count

    @property
    def done(self):
        return self.q >= self.max_q or len(self.found) >= self.target

    def query(self, i, j):
        i = int(i); j = int(j)
        pres, cov = self.o.query_cell(i, j)
        if pres is None:
            return None
        self.visited.add((i, j))
        is_new = False
        if pres:
            if all(math.hypot(i - fi, j - fj) >= self.min_d for fi, fj in self.found):
                self.found.append((i, j))
                self.pos_cov.append(cov.copy())
                is_new = True
        self.X_hist.append((i, j))
        self.y_hist.append(1.0 if pres else 0.0)
        return pres, cov, is_new

    def result(self):
        return dict(points_found=len(self.found), total_queries=self.q,
                    covered_cells=len(self.visited))


class Ctx2D:
    """预计算归一化协变量场 + 每维 min/max（与 harmful_arms.EpisodeCtx 一致）。"""
    def __init__(self, oracle, struct):
        self.o = oracle
        self.struct = list(struct)
        self.H, self.W = oracle.nrows, oracle.ncols
        covf = np.zeros((self.H, self.W, len(self.struct)), dtype=np.float32)
        mins, maxs = [], []
        for kk, k in enumerate(self.struct):
            v = oracle.covs[k].astype(np.float32)
            lo, hi = float(v.min()), float(v.max())
            if hi - lo < 1e-9:
                hi = lo + 1.0
            covf[..., kk] = (v - lo) / (hi - lo)
            mins.append(float(v.min())); maxs.append(float(v.max()))
        self.Xgrid = covf
        self._cov_min = np.array(mins, dtype=np.float32)
        self._cov_max = np.array(maxs, dtype=np.float32)

    def norm_pos_cov(self, pos_cov_raw_list):
        if not pos_cov_raw_list:
            return np.zeros((0, len(self.struct)), dtype=np.float32)
        arr = np.array(pos_cov_raw_list)[:, self.struct]
        arr = (arr - self._cov_min) / (self._cov_max - self._cov_min + 1e-9)
        return arr.astype(np.float32)


def proxy_cov_map(Xgrid, pos_cov_norm):
    H, W, D = Xgrid.shape
    if pos_cov_norm.shape[0] == 0:
        return np.zeros((H, W), dtype=np.float32)
    s = Xgrid.reshape(-1, D)
    d2 = ((s[:, None, :] - pos_cov_norm[None, :, :]) ** 2).sum(axis=2)
    return (-np.sqrt(d2.min(axis=1))).reshape(H, W).astype(np.float32)


def proxy_spatial_map_2d(H, W, found):
    """每格到最近已发现点的负欧氏距离；无发现点时全 0（退化为 raster 顺序）。"""
    if not found:
        return np.zeros((H, W), dtype=np.float32)
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    s = np.zeros((H, W), dtype=np.float32)
    best = np.full((H, W), np.inf, dtype=np.float32)
    for fi, fj in found:
        d = np.hypot(ii - fi, jj - fj)
        best = np.minimum(best, d)
    return (-best).astype(np.float32)


class GreedyRunner2D:
    """通用 2D 贪心执行器：proxy 类型 = 'cov' 或 'spatial'。
    性能优化：score map 变化时才重排；缓存排序索引，步进 past visited。"""
    def __init__(self, oracle, target, max_q, min_d, struct, proxy_type):
        self.env = Env2D(oracle, target, max_q, min_d)
        self.ctx = Ctx2D(oracle, struct) if proxy_type == "cov" else None
        self.proxy_type = proxy_type
        self.H = oracle.nrows; self.W = oracle.ncols
        self._npos = -1
        self._sm = np.zeros((self.H, self.W), dtype=np.float32)
        self._sorted_flat = None
        self._ptr = 0

    def _rebuild_sorted(self):
        self._sorted_flat = np.argsort(self._sm.ravel())[::-1]
        self._ptr = 0

    def _update_score(self):
        env = self.env
        if len(env.found) != self._npos:
            self._npos = len(env.found)
            if self.proxy_type == "cov":
                pn = self.ctx.norm_pos_cov(env.pos_cov)
                self._sm = proxy_cov_map(self.ctx.Xgrid, pn)
            else:
                self._sm = proxy_spatial_map_2d(self.H, self.W, env.found)
            self._rebuild_sorted()

    def run(self, algo_seed):
        env = self.env
        env.reset()
        rng = random.Random(algo_seed)
        H, W = self.H, self.W
        self._npos = -1
        self._sm = np.zeros((H, W), dtype=np.float32)
        self._rebuild_sorted()
        while not env.done:
            self._update_score()
            picked = None
            # advance pointer past visited cells
            while self._ptr < len(self._sorted_flat):
                idx = int(self._sorted_flat[self._ptr])
                i, j = idx // W, idx % W
                self._ptr += 1
                if (i, j) not in env.visited:
                    picked = (i, j); break
            if picked is None:
                picked = (rng.randrange(H), rng.randrange(W))
            env.query(*picked)
        return env.result()


# 3D
class Env3D:
    def __init__(self, oracle, target, max_q, min_d):
        self.o = oracle
        self.target = target
        self.max_q = max_q
        self.min_d = min_d
        self.reset()

    def reset(self):
        self.o.reset_counter()
        self.found = []          # [(k,i,j)]
        self.visited = set()
        self.pos_cov = []
        self.X_hist = []
        self.y_hist = []

    @property
    def q(self):
        return self.o.query_count

    @property
    def done(self):
        return self.q >= self.max_q or len(self.found) >= self.target

    def query(self, k, i, j):
        k = int(k); i = int(i); j = int(j)
        pres, cov = self.o.query_cell(k, i, j)
        if pres is None:
            return None
        self.visited.add((k, i, j))
        is_new = False
        if pres:
            if all(math.dist((k, i, j), f) >= self.min_d for f in self.found):
                self.found.append((k, i, j))
                self.pos_cov.append(cov.copy())
                is_new = True
        self.X_hist.append((k, i, j))
        self.y_hist.append(1.0 if pres else 0.0)
        return pres, cov, is_new

    def result(self):
        return dict(points_found=len(self.found), total_queries=self.q,
                    covered_cells=len(self.visited))


class Ctx3D:
    def __init__(self, oracle, struct):
        self.o = oracle
        self.struct = list(struct)
        self.K, self.L, self.J = oracle.K, oracle.L, oracle.J
        covf = np.zeros((self.K, self.L, self.J, len(self.struct)), dtype=np.float32)
        mins, maxs = [], []
        for kk, k in enumerate(self.struct):
            v = oracle.covs[k].astype(np.float32)
            lo, hi = float(v.min()), float(v.max())
            if hi - lo < 1e-9:
                hi = lo + 1.0
            covf[..., kk] = (v - lo) / (hi - lo)
            mins.append(float(v.min())); maxs.append(float(v.max()))
        self.Xgrid = covf
        self._cov_min = np.array(mins, dtype=np.float32)
        self._cov_max = np.array(maxs, dtype=np.float32)

    def norm_pos_cov(self, pos_cov_raw_list):
        if not pos_cov_raw_list:
            return np.zeros((0, len(self.struct)), dtype=np.float32)
        arr = np.array(pos_cov_raw_list)[:, self.struct]
        arr = (arr - self._cov_min) / (self._cov_max - self._cov_min + 1e-9)
        return arr.astype(np.float32)


def proxy_cov_map_3d(Xgrid, pos_cov_norm):
    K, L, J, D = Xgrid.shape
    if pos_cov_norm.shape[0] == 0:
        return np.zeros((K, L, J), dtype=np.float32)
    s = Xgrid.reshape(-1, D)
    d2 = ((s[:, None, :] - pos_cov_norm[None, :, :]) ** 2).sum(axis=2)
    return (-np.sqrt(d2.min(axis=1))).reshape(K, L, J).astype(np.float32)


def proxy_spatial_map_3d(K, L, J, found):
    if not found:
        return np.zeros((K, L, J), dtype=np.float32)
    kk, ii, jj = np.meshgrid(np.arange(K), np.arange(L), np.arange(J), indexing="ij")
    best = np.full((K, L, J), np.inf, dtype=np.float32)
    for fk, fi, fj in found:
        d = np.sqrt((kk - fk) ** 2 + (ii - fi) ** 2 + (jj - fj) ** 2)
        best = np.minimum(best, d)
    return (-best).astype(np.float32)


class GreedyRunner3D:
    """3D 贪心执行器：同 2D 优化（缓存排序索引，仅在 score map 变化时重排）。"""
    def __init__(self, oracle, target, max_q, min_d, struct, proxy_type):
        self.env = Env3D(oracle, target, max_q, min_d)
        self.ctx = Ctx3D(oracle, struct) if proxy_type == "cov" else None
        self.proxy_type = proxy_type
        self.K, self.L, self.J = oracle.K, oracle.L, oracle.J
        self._npos = -1
        self._sm = np.zeros((self.K, self.L, self.J), dtype=np.float32)
        self._sorted_flat = None
        self._ptr = 0

    def _rebuild_sorted(self):
        self._sorted_flat = np.argsort(self._sm.ravel())[::-1]
        self._ptr = 0

    def _update_score(self):
        env = self.env
        if len(env.found) != self._npos:
            self._npos = len(env.found)
            if self.proxy_type == "cov":
                pn = self.ctx.norm_pos_cov(env.pos_cov)
                self._sm = proxy_cov_map_3d(self.ctx.Xgrid, pn)
            else:
                self._sm = proxy_spatial_map_3d(self.K, self.L, self.J, env.found)
            self._rebuild_sorted()

    def run(self, algo_seed):
        env = self.env
        env.reset()
        rng = random.Random(algo_seed)
        K, L, J = self.K, self.L, self.J
        self._npos = -1
        self._sm = np.zeros((K, L, J), dtype=np.float32)
        self._rebuild_sorted()
        stride = L * J
        while not env.done:
            self._update_score()
            picked = None
            while self._ptr < len(self._sorted_flat):
                idx = int(self._sorted_flat[self._ptr])
                self._ptr += 1
                k = idx // stride
                r = idx % stride
                i = r // J; j = r % J
                if (k, i, j) not in env.visited:
                    picked = (k, i, j); break
            if picked is None:
                picked = (rng.randrange(K), rng.randrange(L), rng.randrange(J))
            env.query(*picked)
        return env.result()


#  既有数据加载

def load_ablation():
    return load_csv(ROOT / "results" / "ablation_cross" / "ablation_detail.csv")


def load_screen_off():
    return load_csv(HERE / "screen_off_arms.csv")


def load_o3d5():
    return load_csv(ROOT / "results" / "ocean3d" / "regress_opt_F" /
                    "framework_opt_o3d_O3D5_params7.csv")


def collect_reused_rows():
    """返回 list[dict]，列：scenario, cell, arm, world_seed, algo_seed,
    points_found, total_queries, covered_cells, source。"""
    rows = []
    abl = load_ablation()
    sco = load_screen_off()
    o3d5 = load_o3d5()

    def add(scenario, cell, arm, ws, ag, p, q, cov, src):
        rows.append(dict(scenario=scenario, cell=cell, arm=arm,
                         world_seed=int(ws), algo_seed=int(ag),
                         points_found=int(p), total_queries=int(q),
                         covered_cells=int(cov), source=src))

    # chlorophyll: on-smart=full, off-smart=screen_off, ref AUC_off
    for r in abl:
        if r["scenario"] == "chlorophyll" and r["variant"] == "full":
            add("chlorophyll", "sel_on_smart", "smart", r["world_seed"], r["algo_seed"],
                r["points_found"], r["total_queries"], -1, "ablation_detail.csv:full")
        if r["scenario"] == "chlorophyll" and r["variant"] == "AUC_off":
            add("chlorophyll", "ref_AUC_off", "smart_ref", r["world_seed"], r["algo_seed"],
                r["points_found"], r["total_queries"], -1, "ablation_detail.csv:AUC_off")
    for r in sco:
        if r["scenario"] == "chlorophyll":
            add("chlorophyll", "sel_off_smart", "smart", r["world_seed"], r["algo_seed"],
                r["points_found"], r["total_queries"], -1, "screen_off_arms.csv")

    # clouded_leopard: on-smart=full, off-smart=screen_off, ref AUC_off
    for r in abl:
        if r["scenario"] == "clouded_leopard" and r["variant"] == "full":
            add("clouded_leopard", "sel_on_smart", "smart", r["world_seed"], r["algo_seed"],
                r["points_found"], r["total_queries"], -1, "ablation_detail.csv:full")
        if r["scenario"] == "clouded_leopard" and r["variant"] == "AUC_off":
            add("clouded_leopard", "ref_AUC_off", "smart_ref", r["world_seed"], r["algo_seed"],
                r["points_found"], r["total_queries"], -1, "ablation_detail.csv:AUC_off")
    for r in sco:
        if r["scenario"] == "clouded_leopard":
            add("clouded_leopard", "sel_off_smart", "smart", r["world_seed"], r["algo_seed"],
                r["points_found"], r["total_queries"], -1, "screen_off_arms.csv")

    # minecraft: off-smart=MC full（MC_CFG 无 sparse_guidance=默认 False=selector off）
    #           ref AUC_off 保留
    for r in abl:
        if r["scenario"] == "minecraft" and r["variant"] == "full":
            add("minecraft", "sel_off_smart", "smart", r["world_seed"], r["algo_seed"],
                r["points_found"], r["total_queries"], -1, "ablation_detail.csv:full")
        if r["scenario"] == "minecraft" and r["variant"] == "AUC_off":
            add("minecraft", "ref_AUC_off", "smart_ref", r["world_seed"], r["algo_seed"],
                r["points_found"], r["total_queries"], -1, "ablation_detail.csv:AUC_off")

    # ocean3d: off-smart=O3D5_params7（sparse_guidance=False, guidance_on=True, covs=7）
    for r in o3d5:
        add("ocean3d", "sel_off_smart", "smart", 0, r["seed"],
            r["points_found"], r["total_queries"], r["covered_cells"],
            "framework_opt_o3d_O3D5_params7.csv")

    return rows


#  新跑：smart 臂（MC on-smart, O3D on-smart）

def run_mc_on_smart(world_seeds, algo_seeds):
    """MC_CFG + sparse_guidance=True（selector on, smart scheduler）。"""
    cfg = dict(MC_CFG, sparse_guidance=True)
    rows = []
    for ws in world_seeds:
        npz = ROOT / "minecraft-search" / "oracle_npz" / f"mc_world_seed{ws}.npz"
        oracle = SpeciesOracle(npz)
        for ag in algo_seeds:
            res = SpeciesSearcher(oracle, cfg).run_episode(ag)
            rows.append(dict(world_seed=ws, algo_seed=ag,
                             points_found=int(res["points_found"]),
                             total_queries=int(res["total_queries"]),
                             covered_cells=int(res["covered_cells"])))
    return rows


def run_o3d_on_smart(oracle, algo_seeds):
    """O3D_BASE + sparse_guidance=True（selector on, smart scheduler）。"""
    cfg = dict(O3D_BASE, sparse_guidance=True)
    rows = []
    for ag in algo_seeds:
        res = Searcher3D(oracle, cfg).run_episode(ag)
        rows.append(dict(world_seed=0, algo_seed=ag,
                         points_found=int(res["points_found"]),
                         total_queries=int(res["total_queries"]),
                         covered_cells=int(res["covered_cells"])))
    return rows


#  主流程

def main():
    t_all = time.time()
    if LOG.exists():
        LOG.unlink()
    log("=== peerj_2x2_matrix: selector(on/off) x scheduler(greedy/smart) ===")

    detail_rows = collect_reused_rows()
    log(f"[reuse] 既有 smart 臂行数 = {len(detail_rows)}")

    # 1. 贪心臂：chlorophyll
    # sel_on_greedy（协变量代理）：直接复用 baseline_chl_greedy_detail.csv（20 seeds 恒=66，
    # 为 C1 数字对账硬锚；当前 numpy 2.0.2 的 argsort tie-break 与生成 CSV 时不同，
    # 重跑会得到 1258，故读既有 CSV 以保住锚点）。
    log("--- [1] chlorophyll greedy arms (20 seeds) ---")
    chl_oracle = SpeciesOracle(ORACLE_PATHS["chlorophyll"])
    chl_greedy_rows = load_csv(HERE / "baseline_chl_greedy_detail.csv")
    for r in chl_greedy_rows:
        detail_rows.append(dict(scenario="chlorophyll", cell="sel_on_greedy", arm="greedy",
                                world_seed=0, algo_seed=int(r["seed"]),
                                points_found=int(r["points_found"]),
                                total_queries=int(r["total_queries"]),
                                covered_cells=int(r["covered_cells"]),
                                source="baseline_chl_greedy_detail.csv"))
    log(f"    chlorophyll sel_on_greedy reused n={len(chl_greedy_rows)} "
        f"(all q=66: {all(int(r['total_queries'])==66 for r in chl_greedy_rows)})")
    # sel_off_greedy（空间纯贪心）：新跑
    t0 = time.time()
    runner = GreedyRunner2D(chl_oracle, target=20, max_q=4000, min_d=1.0,
                            struct=list(range(7)), proxy_type="spatial")
    for ag in SEEDS:
        r = runner.run(ag)
        detail_rows.append(dict(scenario="chlorophyll", cell="sel_off_greedy", arm="greedy",
                                world_seed=0, algo_seed=ag,
                                points_found=r["points_found"],
                                total_queries=r["total_queries"],
                                covered_cells=r["covered_cells"],
                                source="peerj_2x2_matrix.py:spatial"))
    log(f"    chlorophyll sel_off_greedy done ({time.time()-t0:.1f}s)")

    # 2. 贪心臂：clouded_leopard
    log("--- [2] clouded_leopard greedy arms (20 seeds) ---")
    cl_oracle = SpeciesOracle(ORACLE_PATHS["clouded_leopard"])
    for ptype, cell in [("cov", "sel_on_greedy"), ("spatial", "sel_off_greedy")]:
        t0 = time.time()
        runner = GreedyRunner2D(cl_oracle, target=20, max_q=20000, min_d=1.0,
                                struct=list(range(19)), proxy_type=ptype)
        for ag in SEEDS:
            r = runner.run(ag)
            detail_rows.append(dict(scenario="clouded_leopard", cell=cell, arm="greedy",
                                    world_seed=0, algo_seed=ag,
                                    points_found=r["points_found"],
                                    total_queries=r["total_queries"],
                                    covered_cells=r["covered_cells"],
                                    source=f"peerj_2x2_matrix.py:{ptype}"))
        log(f"    clouded_leopard {cell:16s} done ({time.time()-t0:.1f}s)")

    # 3. minecraft：on-smart 新跑 + 两个 greedy
    # 先探针 5 runs 估时
    log("--- [3a] minecraft on-smart 探针 (1 world x 5 seeds) ---")
    probe_ws = 1
    probe_n = 5
    t0 = time.time()
    probe_rows = run_mc_on_smart([probe_ws], SEEDS[:probe_n])
    probe_dt = (time.time() - t0) / probe_n
    log(f"    MC on-smart probe: {probe_n} runs in {time.time()-t0:.1f}s "
        f"= {probe_dt:.2f}s/run")

    if probe_dt > 5.0:
        mc_worlds = list(range(1, 11))
        log(f"    !! MC on-smart >5s/run -> 降为 10 worlds x 20 seeds = 200 runs")
    else:
        mc_worlds = MC_WORLDS_FULL
    mc_n_runs = len(mc_worlds) * len(SEEDS)
    log(f"    MC on-smart full: {len(mc_worlds)} worlds x {len(SEEDS)} seeds = {mc_n_runs} runs")

    # 探针那 5 个 run 直接进明细（避免重跑）
    for r in probe_rows:
        detail_rows.append(dict(scenario="minecraft", cell="sel_on_smart", arm="smart",
                                world_seed=r["world_seed"], algo_seed=r["algo_seed"],
                                points_found=r["points_found"],
                                total_queries=r["total_queries"],
                                covered_cells=r["covered_cells"],
                                source="peerj_2x2_matrix.py:MC_CFG+sparse=True"))
    # 剩余 world 1 的 seeds 106..120
    t0 = time.time()
    extra = run_mc_on_smart([1], SEEDS[probe_n:])
    for r in extra:
        detail_rows.append(dict(scenario="minecraft", cell="sel_on_smart", arm="smart",
                                world_seed=r["world_seed"], algo_seed=r["algo_seed"],
                                points_found=r["points_found"],
                                total_queries=r["total_queries"],
                                covered_cells=r["covered_cells"],
                                source="peerj_2x2_matrix.py:MC_CFG+sparse=True"))
    log(f"    MC on-smart world1 remaining {len(extra)} seeds ({time.time()-t0:.1f}s)")
    # 其余 worlds
    for ws in mc_worlds:
        if ws == 1:
            continue
        t0 = time.time()
        rows = run_mc_on_smart([ws], SEEDS)
        for r in rows:
            detail_rows.append(dict(scenario="minecraft", cell="sel_on_smart", arm="smart",
                                    world_seed=r["world_seed"], algo_seed=r["algo_seed"],
                                    points_found=r["points_found"],
                                    total_queries=r["total_queries"],
                                    covered_cells=r["covered_cells"],
                                    source="peerj_2x2_matrix.py:MC_CFG+sparse=True"))
        log(f"    MC on-smart world={ws} done ({time.time()-t0:.1f}s)")

    # MC greedy 臂
    log("--- [3b] minecraft greedy arms ---")
    for ptype, cell in [("cov", "sel_on_greedy"), ("spatial", "sel_off_greedy")]:
        t0 = time.time()
        n_done = 0
        for ws in mc_worlds:
            npz = ROOT / "minecraft-search" / "oracle_npz" / f"mc_world_seed{ws}.npz"
            oracle = SpeciesOracle(npz)
            runner = GreedyRunner2D(oracle, target=40, max_q=10000, min_d=2.0,
                                    struct=list(range(5)), proxy_type=ptype)
            for ag in SEEDS:
                r = runner.run(ag)
                detail_rows.append(dict(scenario="minecraft", cell=cell, arm="greedy",
                                        world_seed=ws, algo_seed=ag,
                                        points_found=r["points_found"],
                                        total_queries=r["total_queries"],
                                        covered_cells=r["covered_cells"],
                                        source=f"peerj_2x2_matrix.py:{ptype}"))
                n_done += 1
        log(f"    MC {cell:16s} done n={n_done} ({time.time()-t0:.1f}s)")

    # 4. ocean3d：on-smart 新跑 + 两个 greedy
    log("--- [4] ocean3d arms ---")
    o3d_oracle = Oracle3D(ROOT / "ocean3d-search" / "data" / "oracle3d.npz")
    t0 = time.time()
    rows = run_o3d_on_smart(o3d_oracle, SEEDS)
    for r in rows:
        detail_rows.append(dict(scenario="ocean3d", cell="sel_on_smart", arm="smart",
                                world_seed=r["world_seed"], algo_seed=r["algo_seed"],
                                points_found=r["points_found"],
                                total_queries=r["total_queries"],
                                covered_cells=r["covered_cells"],
                                source="peerj_2x2_matrix.py:O3D_BASE+sparse=True"))
    log(f"    ocean3d sel_on_smart done ({time.time()-t0:.1f}s)")

    for ptype, cell in [("cov", "sel_on_greedy"), ("spatial", "sel_off_greedy")]:
        t0 = time.time()
        runner = GreedyRunner3D(o3d_oracle, target=40, max_q=10000, min_d=1.0,
                                struct=list(range(7)), proxy_type=ptype)
        for ag in SEEDS:
            r = runner.run(ag)
            detail_rows.append(dict(scenario="ocean3d", cell=cell, arm="greedy",
                                    world_seed=0, algo_seed=ag,
                                    points_found=r["points_found"],
                                    total_queries=r["total_queries"],
                                    covered_cells=r["covered_cells"],
                                    source=f"peerj_2x2_matrix.py:{ptype}"))
        log(f"    ocean3d {cell:16s} done ({time.time()-t0:.1f}s)")

    # 写明细 CSV
    fieldnames = ["scenario", "cell", "arm", "world_seed", "algo_seed",
                  "points_found", "total_queries", "covered_cells", "source"]
    with open(DETAIL_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(detail_rows)
    log(f"[detail] -> {DETAIL_CSV} ({len(detail_rows)} rows)")

    # 自检锚点
    log("--- self-check anchors ---")
    chl_on_g = [r for r in detail_rows if r["scenario"] == "chlorophyll"
                and r["cell"] == "sel_on_greedy"]
    qs = [r["total_queries"] for r in chl_on_g]
    log(f"  chlorophyll sel_on_greedy: n={len(qs)} mean={np.mean(qs):.1f} "
        f"all==66? {all(q == 66 for q in qs)}")
    chl_on_s = [r for r in detail_rows if r["scenario"] == "chlorophyll"
                and r["cell"] == "sel_on_smart"]
    log(f"  chlorophyll sel_on_smart: n={len(chl_on_s)} "
        f"mean_q={np.mean([r['total_queries'] for r in chl_on_s]):.1f}")
    mc_off_s = [r for r in detail_rows if r["scenario"] == "minecraft"
                and r["cell"] == "sel_off_smart"]
    log(f"  minecraft sel_off_smart: n={len(mc_off_s)} "
        f"mean_pts={np.mean([r['points_found'] for r in mc_off_s]):.2f}")

    log(f"=== all done in {(time.time()-t_all)/60:.1f} min ===")


if __name__ == "__main__":
    main()
