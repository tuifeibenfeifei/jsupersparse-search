# -*- coding: utf-8 -*-
# --- repository root (auto-located) ---
import os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve()
for _ in range(8):
    if (REPO_ROOT / 'configs').is_dir():
        break
    REPO_ROOT = REPO_ROOT.parent
# --------------------------------------
"""perm_control_rerun.py — 置换对照复跑（TABLE XIII 全量统一用）。

背景：手稿 TABLE XIII（COVARIATE PERMUTATION CONTROL ON THE CHLOROPHYLL FIELD）
的实时值（Real = 1605.2、perm7 = 2115.8、perm12 = 1631.2）在仓库中无源数据与
实现脚本。为把整篇论文统一到"当前代码 + 数据可精确复现"的口径，本脚本按论文
V-I 节的描述重建该对照：

  - 对 7 个真实协变量（索引 0–6）的每一个平面做固定的空间置换 → arm "perm7"
  - 对 12 个噪声协变量（索引 7–18）的每一个平面做固定的空间置换 → arm "perm12"
  - "each covariate plane is spatially permuted with its marginals preserved
    exactly and its spatial arrangement destroyed"：逐平面独立随机置换
    （保边际分布、毁空间结构），每个 arm 的置换在全部算法种子上固定；
  - presence 字段保持原样（置换只影响 agent 看到的协变量平面，不动出现场）。

输出：
  supplement_stats/perm_oracles/oracle_perm7.npz / oracle_perm12.npz（置换后的 oracle）
  supplement_stats/perm_control.csv（逐 seed 明细）
用法：
  PYTHONPATH=...\.pylibs  python supplement_stats/perm_control_rerun.py
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(REPO_ROOT)
sys.path.insert(0, str(ROOT / "species-search"))
sys.path.insert(0, str(ROOT / "chlorophyll-search"))
sys.path.insert(0, str(ROOT))

from oracle import SpeciesOracle  # noqa: E402
from species_agent import SpeciesSearcher  # noqa: E402

HERE = ROOT / "supplement_stats"
PERM_DIR = HERE / "perm_oracles"
PERM_DIR.mkdir(exist_ok=True)
ORACLE_PATH = ROOT / "chlorophyll-search" / "data" / "oracle.npz"
OUT_CSV = HERE / "perm_control.csv"

SEEDS = list(range(101, 121))  # 与 TABLE XIII 声明一致（N = 20）

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

# 置换种子（固定、可复现；记录于脚本与补充材料）
PERM_SEED_7 = 202607
PERM_SEED_12 = 202612


def make_permuted_oracle(src_npz: Path, idxs, seed: int, out_npz: Path) -> Path:
    d = np.load(src_npz)
    covs = d["covs"].copy()  # (K, H, W)
    rng = np.random.default_rng(seed)
    for k in idxs:
        plane = covs[k].ravel()
        covs[k] = plane[rng.permutation(plane.size)].reshape(covs[k].shape)
    np.savez(out_npz,
             lon0=d["lon0"], lat0=d["lat0"], cell_deg=d["cell_deg"],
             ncols=d["ncols"], nrows=d["nrows"], presence=d["presence"],
             covs=covs.astype(np.float32), cov_names=d["cov_names"],
             study_poly=d["study_poly"])
    return out_npz


def run_arm(name, npz_path):
    oracle = SpeciesOracle(npz_path)
    rows = []
    t0 = time.time()
    for ag in SEEDS:
        res = SpeciesSearcher(oracle, CHL_CFG).run_episode(ag)
        rows.append((ag, res["points_found"], res["total_queries"]))
    mq = sum(r[2] for r in rows) / len(rows)
    mp = sum(r[1] for r in rows) / len(rows)
    print(f"[{name}] pts={mp:.2f} q={mq:.1f}  ({time.time()-t0:.0f}s, n={len(rows)})")
    return rows


def main():
    # 确认协变量命名（前 7 = 真信号，后 12 = 噪声）
    d0 = np.load(ORACLE_PATH)
    names = [str(x) for x in d0["cov_names"]]
    print("cov_names:", names)

    p7 = make_permuted_oracle(ORACLE_PATH, range(0, 7), PERM_SEED_7,
                              PERM_DIR / "oracle_perm7.npz")
    p12 = make_permuted_oracle(ORACLE_PATH, range(7, 19), PERM_SEED_12,
                               PERM_DIR / "oracle_perm12.npz")
    print("permuted oracles written:", p7.name, p12.name)

    r7 = run_arm("perm7", p7)
    r12 = run_arm("perm12", p12)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "points_found", "total_queries"])
        for ag, p, q in r7:
            w.writerow(["perm7", ag, p, q])
        for ag, p, q in r12:
            w.writerow(["perm12", ag, p, q])
    print("WROTE", OUT_CSV)

    # 与 Real（1596.25，ablation_cross 同批）对比的百分比
    real_q = 1596.25
    for name, rows in (("perm7", r7), ("perm12", r12)):
        mq = sum(r[2] for r in rows) / len(rows)
        print(f"{name}: mean_q={mq:.1f}  change vs real={100*(mq-real_q)/real_q:+.1f}%")


if __name__ == "__main__":
    main()
