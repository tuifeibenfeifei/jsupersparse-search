# -*- coding: utf-8 -*-
import os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve()
for _ in range(8):
    if (REPO_ROOT / 'configs').is_dir():
        break
    REPO_ROOT = REPO_ROOT.parent
"""S3 替身对照：同一两阶段屏幕流程下，精筛统计量 AUC vs 互信息（叶绿素场）。

AUC 版阈值 weak_signal_min_auc=0.52；MI 版分箱 B=8、log2 比特，取同数量 K。
输出屏幕排序质量（Rec@7、真协变量均值秩、噪声最佳秩）与选择集质量（选中真/噪声数）。
"""
import csv
import os
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

OUT = Path(REPO_ROOT / "supplement_stats")
NPZ = Path(REPO_ROOT / "chlorophyll-search/data/oracle.npz")

N_BINS = 8
TRUE_IDS = list(range(7))          # sst_neg..coast_dist 为真信号
N_NOISE = 12
SAMPLE_SIZES = [100, 250, 500, 1000, 2000, 4000, 6000]
SAMPLING_SEEDS = [101, 102, 103, 104, 105]
WEAK_SIG_MIN_AUC = 0.52            # 框架默认（species_agent.py weak_signal_min_auc）
SPARSE_MIN_POS = 5                 # 框架默认：正样本 >=5 才启用稀疏屏幕（sparse_min_pos）
SPARSE_TOPK_BASE = 6               # 框架默认 sparse_topk，随 n_pos 渐进放宽
VARIANCE_TOP = 12                  # 框架默认预筛 top-12

rng_master = np.random.RandomState(2026)
d = np.load(NPZ)
presence = d["presence"].astype(bool)
covs = d["covs"].astype(np.float64)               # (19, H, W)
H, W = presence.shape
all_idx = np.argwhere(presence)                    # (Ncell, 2)
NCELL = presence.size


def auc_separation(pos_vals, neg_vals):
    """复刻框架 _auc：MWU AUC，方向无关（取 max(auc, 1-auc)）"""
    n, m = len(pos_vals), len(neg_vals)
    if n == 0 or m == 0:
        return np.nan
    allv = np.concatenate([pos_vals, neg_vals])
    r = rankdata(allv)
    u = float(r[:n].sum() - n * (n + 1) / 2.0)
    auc = u / (n * m)
    return max(auc, 1.0 - auc)


def mi_binned(vals, y, bins=N_BINS):
    """等宽分箱互信息 I(Cov_bin; Y)，log2 比特"""
    if len(vals) < 2:
        return 0.0
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-12:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    b = np.clip(np.digitize(vals, edges) - 1, 0, bins - 1)
    n = len(vals)
    mi = 0.0
    for bb in range(bins):
        m_b = int((b == bb).sum())
        if m_b == 0:
            continue
        p_b = m_b / n
        y1 = int(y[b == bb].sum())
        p_y1 = y1 / n
        if y1 > 0:
            p1g = y1 / m_b
            mi += p_b * p1g * np.log2(p1g / p_y1) if p_y1 > 0 and p1g > 0 else 0.0
        y0 = m_b - y1
        if y0 > 0:
            p0g = y0 / m_b
            p_y0 = (n - int(y.sum())) / n
            mi += p_b * p0g * np.log2(p0g / p_y0) if p_y0 > 0 and p0g > 0 else 0.0
    return mi


def bin_density_variance(vals, y, bins=N_BINS):
    """复刻框架粗筛统计量（species_agent.py L315-334）：分箱 B=8 → 每箱 presence
    密度 → 密度向量方差；无查询箱用全局出现率平滑。"""
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    b = np.clip(np.digitize(vals, edges) - 1, 0, bins - 1)
    cnt = np.bincount(b, minlength=bins)
    pos = np.bincount(b, weights=y, minlength=bins)
    dens = np.zeros(bins)
    nz = cnt > 0
    dens[nz] = pos[nz] / cnt[nz]
    dens[~nz] = pos.sum() / max(cnt.sum(), 1)
    return float(np.var(dens))


def evaluate(sample_idx, sample_y):
    """给定样本（网格索引 + presence），复刻框架稀疏屏幕流程并替换精筛统计量。

    框架流程（species_agent.py L336-373）：
      触发 n_pos>=5 → 粗筛 top-12（箱密度方差）→ 精筛排序取 topk → 保留 >weak_signal_min_auc
      topk = sparse_topk_base + (n_pos-5)//15，取 min(.., 池大小)
    MI 版：同一粗筛池、同一 topk，按互信息排序取同样数量 K（只换精筛统计量）。
    """
    n = len(sample_idx)
    X = covs[:, sample_idx[:, 0], sample_idx[:, 1]].T          # (n, 19)
    pos = sample_y
    n_pos = int(pos.sum())
    if n_pos < SPARSE_MIN_POS:                                # 稀疏屏幕触发门槛
        return None

    rows = []
    for k in range(19):
        v = X[:, k]
        pv, nv = v[pos], v[~pos]
        rows.append((k, auc_separation(pv, nv), mi_binned(v, pos),
                     bin_density_variance(v, pos)))
    k, aucs, mis, dens_vars = zip(*rows)
    k = list(k); aucs = list(aucs); mis = list(mis); dens_vars = list(dens_vars)

    def rank_by(score):
        order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
        rk = np.empty(len(k), dtype=int)
        rk[order] = np.arange(1, len(k) + 1)
        return rk

    def quality(score, sel_ids):
        rk = rank_by(score)
        true_rk = [rk[i] for i in TRUE_IDS]
        rec7 = sum(1 for i in TRUE_IDS if rk[i] <= 7) / 7.0
        rec12 = sum(1 for i in TRUE_IDS if rk[i] <= 12)
        noise_best = min(rk[i] for i in range(7, 19))
        sel_true = sum(1 for i in sel_ids if i in TRUE_IDS)
        sel_noise = len(sel_ids) - sel_true
        return dict(rec7=rec7, rec12=rec12, true_mean_rank=float(np.mean(true_rk)),
                    noise_best_rank=int(noise_best), sel_true=sel_true, sel_noise=sel_noise)

    # 共同流程：箱密度方差粗筛 top-12 + 渐进 topk
    var_order = np.argsort(-np.asarray(dens_vars, dtype=float), kind="stable")[:VARIANCE_TOP]
    topk = min(SPARSE_TOPK_BASE + max(0, (n_pos - 5) // 15), len(var_order))

    # AUC 版：粗筛池内按 AUC 排序取 topk，保留 > 阈值
    auc_pool = {k[i]: aucs[i] for i in var_order}
    ranked_a = sorted(var_order, key=lambda i: -auc_pool[i])[:topk]
    sel_auc = [i for i in ranked_a if auc_pool[i] > WEAK_SIG_MIN_AUC]
    K = len(sel_auc)

    # MI 版：同一池、同一 topk，按互信息排序取同样数量 K
    mi_pool = {k[i]: mis[i] for i in var_order}
    ranked_m = sorted(var_order, key=lambda i: -mi_pool[i])[:topk]
    sel_mi = ranked_m[:K]

    out = {}
    out["auc"] = quality(aucs, sel_auc)
    out["mi"] = quality(mis, sel_mi)
    out["n_pos"] = n_pos
    out["K"] = K
    out["pool_true"] = sum(1 for i in var_order if i in TRUE_IDS)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    # 预生成采样（各种子独立）
    records = []
    for seed in SAMPLING_SEEDS:
        rng = np.random.RandomState(seed)
        # 均匀采样网格坐标（不替换，保证同点不重复查询）
        cells = np.random.RandomState(seed).choice(NCELL, size=max(SAMPLE_SIZES), replace=False)
        idx = np.array([(c // W, c % W) for c in cells])
        y = presence[idx[:, 0], idx[:, 1]]
        for n in SAMPLE_SIZES:
            r = evaluate(idx[:n], y[:n])
            if r is None:
                records.append(dict(seed=seed, n=n, n_pos=0, K=0, screen="auc",
                                    pool_true=0, rec7="", rec12="", true_mean_rank="",
                                    noise_best_rank="", sel_true="", sel_noise=""))
                continue
            for screen in ("auc", "mi"):
                q = r[screen]
                records.append(dict(seed=seed, n=n, n_pos=r["n_pos"], K=r["K"], screen=screen,
                                    pool_true=r["pool_true"],
                                    rec7=f"{q['rec7']:.3f}", rec12=f"{q['rec12']:.3f}",
                                    true_mean_rank=f"{q['true_mean_rank']:.2f}",
                                    noise_best_rank=q["noise_best_rank"],
                                    sel_true=q["sel_true"], sel_noise=q["sel_noise"]))

    # 写 CSV
    csv_path = OUT / "screen_mi_vs_auc.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    # 汇总（均值；n_pos 取均值）
    print("=" * 92)
    print("S3 信息论屏幕 vs AUC 屏幕（叶绿素 7真+12噪；粗筛 top-12 → 精筛；5 种子均值）")
    print("=" * 92)
    print(f"{'n':>6} {'n_pos':>5} {'池真':>4} {'K':>3} | {'screen':<5} {'Rec@7':>6} {'Rec@12':>6} "
          f"{'真均值秩':>7} {'噪最佳秩':>7} {'选中真':>5} {'选中噪':>5}")
    for n in SAMPLE_SIZES:
        for screen in ("auc", "mi"):
            rs = [r for r in records if r["n"] == n and r["screen"] == screen and r["rec7"] != ""]
            if not rs:
                print(f"{n:>6} {'—':>5} {'—':>4} {'—':>3} | {screen:<5}  (n_pos<5，稀疏屏幕不触发)")
                continue
            def avg(field):
                vals = [float(r[field]) for r in rs]
                return sum(vals) / len(vals)
            n_pos = int(round(avg("n_pos")))
            pool_t = int(round(avg("pool_true")))
            kk = int(round(avg("K")))
            print(f"{n:>6} {n_pos:>5} {pool_t:>4} {kk:>3} | {screen:<5} {avg('rec7'):>6.3f} {avg('rec12'):>6.3f} "
                  f"{avg('true_mean_rank'):>7.2f} {avg('noise_best_rank'):>7.0f} "
                  f"{avg('sel_true'):>5.1f} {avg('sel_noise'):>5.1f}")
    print("\nCSV:", csv_path)


if __name__ == "__main__":
    main()
