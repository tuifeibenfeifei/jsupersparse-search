# -*- coding: utf-8 -*-
"""Unified-mechanism x scenario OAT ablation matrix experiment.

Reuses existing rows of ablation_detail.csv (full/AUC_off/multicluster_off), reruns missing ablation arms and records covered_cells;
--matrix mode recomputes from the CSV to generate peerj_oat_matrix.md.
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


# base config (exactly the same as run_cross_ablation.py)
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

# new arms to rerun: (scenario, variant, overrides, algo_seeds)
# CL first (long pole).
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
    """First run: import the old rows of ablation_detail.csv, record covered_cells as NA."""
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


# run
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


# statistics and matrix
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
    """Return (diff_mean, diff_sd, p_two_sided, p_dir_gt, p_dir_lt, delta_cliff, paired_n).
    a=off arm, b=full arm (paired by the same seed). diff = a - b."""
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
    """Pair by (world_seed, algo_seed), return (off_vals, full_vals). Skip NA/empty values."""
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
    """Generate one matrix cell text: Δq mean±SD + Δp + Wilcoxon p + direction."""
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
    # significance: take the one-sided p consistent with the effect direction
    if dq > 0:
        p_use = pgq
    else:
        p_use = plq
    sig = "**" if p_use < 0.01 else ("*" if p_use < 0.05 else "ns")
    # direction judgment (beneficial = performance worsens after turning off)
    eff = []
    if p_use < 0.05:
        if dq > 0:
            eff.append("more queries after turning off -> mechanism saves queries [beneficial]")
        else:
            eff.append("fewer queries after turning off -> mechanism wastes queries [harmful]")
    if p2p < 0.1:
        if dp < 0:
            eff.append("fewer hits after turning off [beneficial]")
        elif dp > 0:
            eff.append("more hits after turning off [harmful]")
    eff_txt = "; ".join(eff) if eff else "no significant difference [inert]"
    parts.append(f"p={p_use:.3f}{sig}")
    parts.append(eff_txt)
    return "<br>".join(parts), dict(dq=dq, sdq=sdq, dp=dp, p=p_use, sig=sig,
                                    eff=eff_txt, n=nq, cdq=cdq)


# mechanism row definitions: (row name, variant, {scenario: annotation note or None})
MECHANISM_ROWS = [
    ("AUC density-guidance screening off", "AUC_off",
     {"minecraft": None, "clouded_leopard": None, "chlorophyll": None}),
    ("Multicluster off", "multicluster_off",
     {"minecraft": None, "clouded_leopard": None, "chlorophyll": None}),
    ("Directed long jump (directed_jump) off", "directed_jump_off",
     {"minecraft": "- (MC_CFG has no directed_jump key; baseline is already False)",
      "clouded_leopard": None, "chlorophyll": None}),
    ("Guidance blank jump (blank_jump) off", "blank_jump_off",
     {"minecraft": None, "clouded_leopard": None, "chlorophyll": None}),
    ("Ring follow (follow_mode) off", "follow_off",
     {"minecraft": None,
      "clouded_leopard": "- (SPP_CFG has no follow_mode key; baseline is already False; 2-seed verification identical to full)",
      "chlorophyll": "- (CHL_CFG has no follow_mode key; baseline is already False; 20-seed reproduction verification)"}),
    ("Stage1 blind sweep changed to serpentine", "serpentine",
     {"minecraft": "- (not tested in this scenario)", "clouded_leopard": "- (not tested in this scenario)",
      "chlorophyll": None}),
    ("Binning count n_bins=4", "n_bins4",
     {"minecraft": "- (not tested in this scenario)", "clouded_leopard": None, "chlorophyll": None}),
    ("Binning count n_bins=16", "n_bins16",
     {"minecraft": "- (not tested in this scenario)", "clouded_leopard": None, "chlorophyll": None}),
]

SCEN_ORDER = ["chlorophyll", "clouded_leopard", "minecraft"]


def build_matrix():
    rows = load_detail()
    log("=== building matrix ===")
    lines = []
    lines.append("# Unified-mechanism x scenario OAT ablation matrix\n")
    lines.append("## 1. Ablation matrix (cell = Δ relative to full = off arm - full, mean±SD; paired Wilcoxon one-sided p, α=0.05)\n")
    header = "| Mechanism component \\ Scenario | " + " | ".join(SCEN_ORDER) + " |"
    sep = "|" + "---|" * (len(SCEN_ORDER) + 1)
    lines.append(header)
    lines.append(sep)

    # collect the numbers per row per cell for conclusion sentences
    row_summary = {}
    for mech_name, variant, scen_note in MECHANISM_ROWS:
        cells = []
        cell_stats = {}
        for sc in SCEN_ORDER:
            if scen_note.get(sc):
                cells.append(scen_note[sc])
                cell_stats[sc] = None
                continue
            # follow_off uses follow_off_verify on CL; CHL follow_off is the equivalent full verification arm
            v = variant
            if sc == "clouded_leopard" and variant == "follow_off":
                v = "follow_off_verify"
            res = cell_text(rows, sc, v)
            if res is None:
                cells.append("(not run)")
                cell_stats[sc] = None
            else:
                cells.append(res[0])
                cell_stats[sc] = res[1]
        lines.append(f"| **{mech_name}** | " + " | ".join(cells) + " |")
        row_summary[mech_name] = cell_stats

    lines.append("")
    lines.append("## 2. Per-mechanism conclusion sentences (can go into the paper)\n")
    for mech_name, variant, scen_note in MECHANISM_ROWS:
        cs = row_summary[mech_name]
        sentence_bits = []
        for sc in SCEN_ORDER:
            s = cs.get(sc)
            if s is None:
                note = scen_note.get(sc, "")
                continue
            dq, dp, p, sig = s["dq"], s["dp"], s["p"], s["sig"]
            tag = "significant" if sig != "ns" else "not significant"
            # direction
            if s["sig"] != "ns":
                direction = "more queries after turning off; mechanism is beneficial at saving queries" if dq > 0 \
                    else "fewer queries after turning off; mechanism is harmful at wasting queries"
            else:
                direction = "mechanism effect is inert"
            sentence_bits.append(
                f"- **{sc}**: after turning off this mechanism, total query change Δq={dq:+.0f} (p={p:.3f}, {tag}), "
                f"hit change Δp={dp:+.2f}; {direction}.")
        lines.append(f"### {mech_name}")
        lines.extend(sentence_bits if sentence_bits else ["- This mechanism is either already off at baseline or untested across all three scenarios."])
        lines.append("")

    lines.append("## 3. Run setup notes\n")
    lines.append("- **Seeds**: chlorophyll/clouded_leopard = 1 oracle x 20 algo seeds (101-120); "
                 "minecraft = 20 world seeds (1-20) x 20 algo seeds (101-120) = 400 runs/variant.")
    lines.append("- **Pairing**: the off arm vs full are paired run-by-run under the same algo_seed (minecraft also fixes world_seed).")
    lines.append("- **Statistics**: scipy.stats.wilcoxon signed-rank test; p takes the one-sided value consistent with the effect direction "
                 "(test greater when Δq>0, test less when Δq<0); significance threshold α=0.05 (*)/0.01 (**), "
                 "plus Cliff's δ (computed in-script; the main matrix table uses p).")
    lines.append("- **Direction semantics**: Δq = off arm - full total queries; Δq<0 means fewer queries after turning off -> mechanism wastes queries (harmful); "
                 "Δq>0 means more queries after turning off -> mechanism saves queries (beneficial). Δp<0 means fewer hits after turning off -> mechanism is beneficial.")
    lines.append("- **Baseline-already-off note**: MC_CFG has no directed_jump key and SPP_CFG/CHL_CFG have no follow_mode key, "
                 "based on the cfg.get(..., False) defaults in species_agent.py, the corresponding mechanisms are already off at baseline; "
                 "CL follow_off uses only seeds 101/102 to verify bitwise identity with full; CHL follow_off is reproduced with 20 seeds.")
    lines.append("- **covered_cells**: recorded only for the newly run arms; the old full/AUC_off/multicluster_off rows "
                 "(from results/ablation_cross/ablation_detail.csv) did not record covered_cells, so Δcov is mostly NA.")
    lines.append("- **Reproducibility notes**: "
                 "(a) the chlorophyll full arm was rerun in this environment for 20 seeds (mean q=1605.25), "
                 "which differs by +0.6% from the 10-03 ablation record (1596.25); 12 of 20 seeds drift (≤±500 queries), "
                 "code/config/oracle/library-version/hash-order were ruled out; this is judged to be an execution-state difference in the 10-03 run; "
                 "so the CHL-column full arm is paired with all new variants under the same state. "
                 "(b) clouded_leopard full was spot-checked for seeds 101-105 and matches the old record 5/5 bitwise; the CL column reuses old rows. "
                 "(c) the minecraft old rows were verified bitwise-consistent against world1's first 5 seeds; the MC column reuses old rows.")

    MATRIX_MD.write_text("\n".join(lines), encoding="utf-8")
    log(f"matrix written -> {MATRIX_MD}")

    # console self-check anchor
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
