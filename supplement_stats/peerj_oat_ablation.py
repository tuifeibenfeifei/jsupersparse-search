# -*- coding: utf-8 -*-
"""统一机制 × 场景 OAT 消融矩阵实验。

复用 ablation_detail.csv 已有行（full/AUC_off/multicluster_off），补跑缺失消融臂并记录 covered_cells；
--matrix 模式从 CSV 重算生成 peerj_oat_matrix.md。
"""
import csv
import sys
import time
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(r"C:\Users\19264\Desktop\deep-sea-exploration")
sys.path.insert(0, str(ROOT / "species-search"))
sys.path.insert(0, str(ROOT / "chlorophyll-search"))

from oracle import SpeciesOracle  # noqa: E402
from species_agent import SpeciesSearcher  # noqa: E402

HERE = Path(__file__).resolve().parent
DETAIL_CSV = HERE / "peerj_oat_detail.csv"
LOG = HERE / "peerj_oat_ablation.log"
MATRIX_MD = HERE / "peerj_oat_matrix.md"
OLD_CSV = ROOT / "results" / "ablation_cross" / "ablation_detail.csv"

SEEDS = list(range(101, 121))            # algo seeds 101..120
MC_WORLD_SEEDS = list(range(1, 21))      # minecraft world seeds 1..20


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# 基础配置（与 run_cross_ablation.py 完全一致）
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

BASE_CFG = {"minecraft": MC_CFG, "clouded_leopard": SPP_CFG, "chlorophyll": CHL_CFG}
ORACLE_PATHS = {
    "clouded_leopard": ROOT / "species-search" / "data" / "clouded_leopard" / "oracle.npz",
    "chlorophyll": ROOT / "chlorophyll-search" / "data" / "oracle.npz",
}

# 待补跑新臂：(scenario, variant, overrides, algo_seeds)
# CL 长杆先跑。
NEW_ARMS = [
    ("clouded_leopard", "blank_jump_off",      {"blank_jump": False},            SEEDS),
    ("clouded_leopard", "directed_jump_off",  {"directed_jump": False},          SEEDS),
    ("clouded_leopard", "n_bins4",             {"n_bins": 4},                     SEEDS),
    ("clouded_leopard", "n_bins16",            {"n_bins": 16},                    SEEDS),
    ("clouded_leopard", "follow_off_verify",   {"follow_mode": False},            [101, 102]),
    ("minecraft",       "follow_off",          {"follow_mode": False},            SEEDS),
    ("minecraft",       "blank_jump_off",      {"blank_jump": False},            SEEDS),
    ("chlorophyll",     "follow_off",          {"follow_mode": False},            SEEDS),
    ("chlorophyll",     "blank_jump_off",      {"blank_jump": False},            SEEDS),
    ("chlorophyll",     "directed_jump_off",  {"directed_jump": False},          SEEDS),
    ("chlorophyll",     "n_bins4",             {"n_bins": 4},                     SEEDS),
    ("chlorophyll",     "n_bins16",            {"n_bins": 16},                    SEEDS),
    ("chlorophyll",     "serpentine",          {"stage1_scan_mode": "serpentine"}, SEEDS),
]


# I/O
HEADER = ["scenario", "variant", "world_seed", "algo_seed",
          "points_found", "total_queries", "covered_cells"]


def load_detail():
    if not DETAIL_CSV.exists():
        return []
    with open(DETAIL_CSV, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def seed_from_old():
    """首次运行：把 ablation_detail.csv 旧行导入，covered_cells 记 NA。"""
    rows = []
    with open(OLD_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({
                "scenario": r["scenario"], "variant": r["variant"],
                "world_seed": r["world_seed"], "algo_seed": r["algo_seed"],
                "points_found": r["points_found"], "total_queries": r["total_queries"],
                "covered_cells": "NA",
            })
    with open(DETAIL_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)
    log(f"seeded {len(rows)} old rows from {OLD_CSV.name} -> {DETAIL_CSV.name}")


def append_row(row):
    with open(DETAIL_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writerow(row)


# 运行
def expected_count(scenario, algo_seeds):
    n_world = len(MC_WORLD_SEEDS) if scenario == "minecraft" else 1
    return n_world * len(algo_seeds)


def existing_count(rows, scenario, variant):
    return sum(1 for r in rows if r["scenario"] == scenario and r["variant"] == variant)


def run_arm(scenario, variant, overrides, algo_seeds):
    cfg = dict(BASE_CFG[scenario], **overrides)
    t0 = time.time()
    new_rows = []
    if scenario == "minecraft":
        for ws in MC_WORLD_SEEDS:
            oracle = SpeciesOracle(ROOT / "minecraft-search" / "oracle_npz"
                                   / f"mc_world_seed{ws}.npz")
            for ag in algo_seeds:
                res = SpeciesSearcher(oracle, cfg).run_episode(ag)
                row = {"scenario": scenario, "variant": variant, "world_seed": ws,
                       "algo_seed": ag,
                       "points_found": res["points_found"],
                       "total_queries": res["total_queries"],
                       "covered_cells": res["covered_cells"]}
                append_row(row)
                new_rows.append(row)
    else:
        oracle = SpeciesOracle(ORACLE_PATHS[scenario])
        for ag in algo_seeds:
            res = SpeciesSearcher(oracle, cfg).run_episode(ag)
            row = {"scenario": scenario, "variant": variant, "world_seed": 0,
                   "algo_seed": ag,
                   "points_found": res["points_found"],
                   "total_queries": res["total_queries"],
                   "covered_cells": res["covered_cells"]}
            append_row(row)
            new_rows.append(row)
    mp = np.mean([r["points_found"] for r in new_rows])
    mq = np.mean([r["total_queries"] for r in new_rows])
    mc = np.mean([r["covered_cells"] for r in new_rows])
    dt = time.time() - t0
    log(f"  DONE {scenario:16s} {variant:20s} pts={mp:6.2f} q={mq:8.1f} "
        f"cov={mc:8.1f}  ({dt:.1f}s, n={len(new_rows)})")
    return new_rows


def run_all():
    if not DETAIL_CSV.exists():
        seed_from_old()
    rows = load_detail()
    log("=== OAT ablation run start ===")
    t_all = time.time()
    for scenario, variant, overrides, algo_seeds in NEW_ARMS:
        exp = expected_count(scenario, algo_seeds)
        have = existing_count(rows, scenario, variant)
        if have >= exp:
            log(f"  SKIP {scenario:16s} {variant:20s} already {have}/{exp}")
            continue
        log(f"  RUN  {scenario:16s} {variant:20s} (have {have}/{exp}, overrides={overrides})")
        run_arm(scenario, variant, overrides, algo_seeds)
        rows = load_detail()
    log(f"=== run all done, total {(time.time()-t_all)/60:.1f} min ===")


# 统计与矩阵
def cliff_delta(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("nan")
    sgn = 0.0
    for x in a:
        sgn += np.sum(b < x) - np.sum(b > x)
    return sgn / (n * m)


def one_sided_paired(a, b):
    """返回 (diff_mean, diff_sd, p_two_sided, p_dir_gt, p_dir_lt, delta_cliff, paired_n).
    a=off 臂, b=full 臂（同 seed 配对）。diff = a - b。"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    d = a - b
    if np.allclose(d, 0):
        return float(np.mean(d)), float(np.std(d, ddof=1)) if n > 1 else 0.0, 1.0, 1.0, 1.0, 0.0, n
    p2 = stats.wilcoxon(d, zero_method="wilcox").pvalue
    try:
        pg = stats.wilcoxon(d, zero_method="wilcox", alternative="greater").pvalue
    except ValueError:
        pg = 1.0
    try:
        pl = stats.wilcoxon(d, zero_method="wilcox", alternative="less").pvalue
    except ValueError:
        pl = 1.0
    cd = cliff_delta(a, b)
    return float(np.mean(d)), float(np.std(d, ddof=1)), float(p2), float(pg), float(pl), float(cd), n


def group_by(rows, scenario, variant):
    return [r for r in rows if r["scenario"] == scenario and r["variant"] == variant]


def pair_vectors(off_rows, full_rows, metric):
    """按 (world_seed, algo_seed) 配对，返回 (off_vals, full_vals)。跳过 NA/空值。"""
    key = lambda r: (int(r["world_seed"]), int(r["algo_seed"]))

    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    fmap = {}
    for r in full_rows:
        val = _f(r[metric])
        if val is not None:
            fmap[key(r)] = val
    ov, fv = [], []
    for r in off_rows:
        k = key(r)
        oval = _f(r[metric])
        if k in fmap and oval is not None:
            ov.append(oval); fv.append(fmap[k])
    return ov, fv


def cell_text(rows, scenario, variant):
    """生成一个矩阵单元格文本：Δq mean±SD + Δp + Wilcoxon p + 方向。"""
    off = group_by(rows, scenario, variant)
    full = group_by(rows, scenario, "full")
    if not off:
        return None
    ov_q, fv_q = pair_vectors(off, full, "total_queries")
    ov_p, fv_p = pair_vectors(off, full, "points_found")
    ov_c, fv_c = pair_vectors(off, full, "covered_cells")
    dq, sdq, p2q, pgq, plq, cdq, nq = one_sided_paired(ov_q, fv_q)
    dp, sdp, p2p, pgp, plp, cdp, np_ = one_sided_paired(ov_p, fv_p)
    parts = [f"Δq={dq:+.0f}±{sdq:.0f}", f"Δp={dp:+.2f}"]
    if ov_c and fv_c:
        dc = np.mean(np.asarray(ov_c) - np.asarray(fv_c))
        parts.append(f"Δcov={dc:+.0f}")
    else:
        parts.append("Δcov=NA")
    # 显著性：取方向一致的单尾 p
    if dq > 0:
        p_use = pgq
    else:
        p_use = plq
    sig = "**" if p_use < 0.01 else ("*" if p_use < 0.05 else "ns")
    # 方向判定（有益=关闭后变差）
    eff = []
    if p_use < 0.05:
        if dq > 0:
            eff.append("关闭后更耗查询→机制省查询[有益]")
        else:
            eff.append("关闭后更省查询→机制费查询[有害]")
    if p2p < 0.1:
        if dp < 0:
            eff.append("关闭后命中下降[有益]")
        elif dp > 0:
            eff.append("关闭后命中上升[有害]")
    eff_txt = "; ".join(eff) if eff else "无显著差异[惰性]"
    parts.append(f"p={p_use:.3f}{sig}")
    parts.append(eff_txt)
    return "<br>".join(parts), dict(dq=dq, sdq=sdq, dp=dp, p=p_use, sig=sig,
                                    eff=eff_txt, n=nq, cdq=cdq)


# 机制行定义：(行名, variant, {scenario: 标注说明 or None})
MECHANISM_ROWS = [
    ("AUC 密度引导筛选关", "AUC_off",
     {"minecraft": None, "clouded_leopard": None, "chlorophyll": None}),
    ("多簇（multicluster）关", "multicluster_off",
     {"minecraft": None, "clouded_leopard": None, "chlorophyll": None}),
    ("定向远跳（directed_jump）关", "directed_jump_off",
     {"minecraft": "—（MC_CFG 无 directed_jump 键，基线已 False）",
      "clouded_leopard": None, "chlorophyll": None}),
    ("导向空白跳（blank_jump）关", "blank_jump_off",
     {"minecraft": None, "clouded_leopard": None, "chlorophyll": None}),
    ("环状跟随（follow_mode）关", "follow_off",
     {"minecraft": None,
      "clouded_leopard": "—（SPP_CFG 无 follow_mode 键，基线已 False；2-seed 验证≡full）",
      "chlorophyll": "—（CHL_CFG 无 follow_mode 键，基线已 False；20-seed 复现验证）"}),
    ("Stage1 盲扫改 serpentine", "serpentine",
     {"minecraft": "—（未在此场景测试）", "clouded_leopard": "—（未在此场景测试）",
      "chlorophyll": None}),
    ("分箱数 n_bins=4", "n_bins4",
     {"minecraft": "—（未在此场景测试）", "clouded_leopard": None, "chlorophyll": None}),
    ("分箱数 n_bins=16", "n_bins16",
     {"minecraft": "—（未在此场景测试）", "clouded_leopard": None, "chlorophyll": None}),
]

SCEN_ORDER = ["chlorophyll", "clouded_leopard", "minecraft"]


def build_matrix():
    rows = load_detail()
    log("=== building matrix ===")
    lines = []
    lines.append("# 统一机制 × 场景 OAT 消融矩阵\n")
    lines.append("## 1. 消融矩阵（单元格 = 相对 full 的 Δ = 关闭臂 − full，mean±SD；配对 Wilcoxon 单尾 p，α=0.05）\n")
    header = "| 机制组件 \\ 场景 | " + " | ".join(SCEN_ORDER) + " |"
    sep = "|" + "---|" * (len(SCEN_ORDER) + 1)
    lines.append(header)
    lines.append(sep)

    # 收集每行每格的数值用于结论句
    row_summary = {}
    for mech_name, variant, scen_note in MECHANISM_ROWS:
        cells = []
        cell_stats = {}
        for sc in SCEN_ORDER:
            if scen_note.get(sc):
                cells.append(scen_note[sc])
                cell_stats[sc] = None
                continue
            # follow_off 在 CL 用 follow_off_verify；CHL follow_off 即等价 full 验证臂
            v = variant
            if sc == "clouded_leopard" and variant == "follow_off":
                v = "follow_off_verify"
            res = cell_text(rows, sc, v)
            if res is None:
                cells.append("（未运行）")
                cell_stats[sc] = None
            else:
                cells.append(res[0])
                cell_stats[sc] = res[1]
        lines.append(f"| **{mech_name}** | " + " | ".join(cells) + " |")
        row_summary[mech_name] = cell_stats

    lines.append("")
    lines.append("## 2. 逐机制结论句（可写入论文）\n")
    for mech_name, variant, scen_note in MECHANISM_ROWS:
        cs = row_summary[mech_name]
        sentence_bits = []
        for sc in SCEN_ORDER:
            s = cs.get(sc)
            if s is None:
                note = scen_note.get(sc, "")
                continue
            dq, dp, p, sig = s["dq"], s["dp"], s["p"], s["sig"]
            tag = "显著" if sig != "ns" else "不显著"
            # 方向
            if s["sig"] != "ns":
                direction = "关闭后查询增多、机制在节省查询上有益" if dq > 0 \
                    else "关闭后查询减少、机制在浪费查询上有害"
            else:
                direction = "机制影响惰性"
            sentence_bits.append(
                f"- **{sc}**：关闭该机制后总查询变化 Δq={dq:+.0f}（p={p:.3f}，{tag}），"
                f"命中变化 Δp={dp:+.2f}；{direction}。")
        lines.append(f"### {mech_name}")
        lines.extend(sentence_bits if sentence_bits else ["- 该机制在三个场景均为基线已关闭/未测试。"])
        lines.append("")

    lines.append("## 3. 运行口径说明\n")
    lines.append("- **Seeds**：chlorophyll/clouded_leopard = 1 oracle × 20 algo seeds（101–120）；"
                 "minecraft = 20 world seeds（1–20）× 20 algo seeds（101–120）= 400 runs/变体。")
    lines.append("- **配对方式**：同一 algo_seed（minecraft 同时固定 world_seed）下关闭臂 vs full 逐 run 配对。")
    lines.append("- **统计**：scipy.stats.wilcoxon 符号秩检验；p 取效应方向一致的单尾值"
                 "（Δq>0 时检验 greater，Δq<0 时检验 less），显著性阈值 α=0.05（*）/0.01（**），"
                 "另附 Cliff's δ（脚本内计算，矩阵主表以 p 为准）。")
    lines.append("- **方向语义**：Δq=关闭臂−full 总查询；Δq<0 表示关闭后更省查询→机制费查询（有害）；"
                 "Δq>0 表示关闭后更费查询→机制省查询（有益）。Δp<0 表示关闭后命中变少→机制有益。")
    lines.append("- **基线已关说明**：MC_CFG 无 directed_jump 键、SPP_CFG/CHL_CFG 无 follow_mode 键，"
                 "依据 species_agent.py 中 cfg.get(..., False) 的缺省值，对应机制在基线已关闭；"
                 "CL follow_off 仅用 seed 101/102 验证与 full 逐位一致，CHL follow_off 用 20 seeds 复现验证。")
    lines.append("- **covered_cells**：仅本次新跑臂记录；旧 full/AUC_off/multicluster_off 行"
                 "（来自 results/ablation_cross/ablation_detail.csv）未记录 covered_cells，故 Δcov 多为 NA。")
    lines.append("- **复现性口径**："
                 "(a) chlorophyll 的 full 臂为本环境重跑 20 seeds（mean q=1605.25），"
                 "与 10-03 ablation 记录（1596.25）相差 +0.6%，20 seeds 中 12 个漂移（≤±500 查询），"
                 "已排除代码/配置/oracle/库版本/哈希序，判定为 10-03 运行的执行状态差异；"
                 "故 CHL 列 full 与全部新变体同状态配对。"
                 "(b) clouded_leopard full 抽查 seed 101–105 与旧记录 5/5 逐位一致，CL 列复用旧行。"
                 "(c) minecraft 旧行经 world1 前 5 seed 核对逐位一致，MC 列复用旧行。")

    MATRIX_MD.write_text("\n".join(lines), encoding="utf-8")
    log(f"matrix written -> {MATRIX_MD}")

    # 控制台自检锚点
    log("=== anchor check ===")
    for sc, var, q_anchor, p_anchor in [
        ("chlorophyll", "full", 1596.25, None),
        ("clouded_leopard", "full", None, 14.00),
        ("minecraft", "full", 3905.0, 29.8),
    ]:
        g = group_by(rows, sc, var)
        mq = np.mean([float(r["total_queries"]) for r in g])
        mp = np.mean([float(r["points_found"]) for r in g])
        log(f"  anchor {sc:16s} {var:8s} q={mq:.1f} (expect {q_anchor}) pts={mp:.2f} (expect {p_anchor}) n={len(g)}")


if __name__ == "__main__":
    if "--matrix" in sys.argv:
        build_matrix()
    else:
        run_all()
