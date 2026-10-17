# -*- coding: utf-8 -*-
"""复现性口径修正后处理（在 peerj_oat_ablation.py 主跑批完成后运行）。

1) clouded_leopard full：seed 101-105 本环境重跑并与 ablation_detail.csv 逐位比对，5/5 一致则复用旧行，否则重跑 20 seeds 替换；
2) chlorophyll full：本环境重跑 20 seeds 替换旧 CHL full 行；
3) minecraft：不处理（旧行已逐位核对）。
"""
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(r"C:\Users\19264\Desktop\deep-sea-exploration")
sys.path.insert(0, str(ROOT / "species-search"))
sys.path.insert(0, str(ROOT / "chlorophyll-search"))
from oracle import SpeciesOracle  # noqa: E402
from species_agent import SpeciesSearcher  # noqa: E402

HERE = Path(__file__).resolve().parent
DETAIL_CSV = HERE / "peerj_oat_detail.csv"
LOG = HERE / "peerj_oat_post.log"
OLD_CSV = ROOT / "results" / "ablation_cross" / "ablation_detail.csv"

# 复用主脚本的配置
import importlib.util
spec = importlib.util.spec_from_file_location(
    "oat", HERE / "peerj_oat_ablation.py")
oat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oat)

SEEDS = list(range(101, 121))


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_rows():
    with open(DETAIL_CSV, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f)), csv.DictReader(open(DETAIL_CSV, encoding="utf-8-sig")).fieldnames


def write_rows(rows, fields):
    with open(DETAIL_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run_full(scenario, algo_seeds):
    cfg = dict(oat.BASE_CFG[scenario])
    out = []
    if scenario == "minecraft":
        raise RuntimeError("MC not handled here")
    oracle = SpeciesOracle(oat.ORACLE_PATHS[scenario])
    for ag in algo_seeds:
        res = SpeciesSearcher(oracle, cfg).run_episode(ag)
        out.append({"scenario": scenario, "variant": "full", "world_seed": 0,
                    "algo_seed": ag,
                    "points_found": res["points_found"],
                    "total_queries": res["total_queries"],
                    "covered_cells": res["covered_cells"]})
    return out


def main():
    log("=== post repro-calibration start ===")
    rows, fields = load_rows()

    # 1) CL spot check seeds 101-105
    with open(OLD_CSV, encoding="utf-8-sig") as f:
        old = list(csv.DictReader(f))
    old_cl_full = {int(r["algo_seed"]): r for r in old
                   if r["scenario"] == "clouded_leopard" and r["variant"] == "full"}
    cl_spot = run_full("clouded_leopard", [101, 102, 103, 104, 105])
    cl_match = 0
    for r in cl_spot:
        o = old_cl_full[r["algo_seed"]]
        same = (int(o["points_found"]) == r["points_found"]
                and int(o["total_queries"]) == r["total_queries"])
        log(f"  CL spot seed {r['algo_seed']}: old(pts={o['points_found']},q={o['total_queries']}) "
            f"new(pts={r['points_found']},q={r['total_queries']}) -> {'MATCH' if same else 'DIFF'}")
        cl_match += int(same)
    log(f"  CL spot-check: {cl_match}/5 identical")

    if cl_match == 5:
        log("  -> CL column keeps old rows (5/5 spot-checked identical).")
        cl_replace = False
    else:
        log("  -> CL full rerun needed; replacing old CL full rows with fresh 20-seed run.")
        cl_fresh = run_full("clouded_leopard", SEEDS)
        rows = [r for r in rows
                if not (r["scenario"] == "clouded_leopard" and r["variant"] == "full")]
        rows.extend(cl_fresh)
        cl_replace = True

    # 2) CHL full rerun 20 seeds
    chl_fresh = run_full("chlorophyll", SEEDS)
    mp = sum(r["points_found"] for r in chl_fresh) / 20
    mq = sum(r["total_queries"] for r in chl_fresh) / 20
    log(f"  CHL full rerun: mean pts={mp:.2f} mean q={mq:.2f} (old 10-03 record q=1596.25)")
    rows = [r for r in rows
            if not (r["scenario"] == "chlorophyll" and r["variant"] == "full")]
    rows.extend(chl_fresh)

    write_rows(rows, fields)
    log(f"  wrote {len(rows)} rows -> {DETAIL_CSV.name}")
    log(f"  CL_replaced={cl_replace}  CHL_fresh_mean_q={mq:.2f}")
    log("=== post done ===")


if __name__ == "__main__":
    main()
