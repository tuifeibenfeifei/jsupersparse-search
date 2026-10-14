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
"""extract_feature_evolution.py — 从特征诊断明细生成演化数据
读 framework_species_sparse_features_<sp>.csv，对指定 seed 按查询区间分桶，
统计每个特征在各区间的平均权重，输出演化序列 CSV（供 ECharts 画收敛曲线）。
"""
import csv
import io
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(REPO_ROOT / "results/species_search")


def parse_feats(s):
    """'bio11=0.353;bio6=0.304;...' -> {name: weight}"""
    out = {}
    for item in s.split(";"):
        if not item:
            continue
        name, w = item.split("=")
        out[name] = float(w)
    return out


def main():
    if len(sys.argv) < 4:
        print("usage: extract_feature_evolution.py <species> <seed> <bucket_queries>")
        return 1
    species = sys.argv[1]
    seed = int(sys.argv[2])
    bucket = int(sys.argv[3])

    rows = []
    with open(OUT / f"framework_species_sparse_features_{species}.csv",
              encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["seed"]) == seed:
                rows.append((int(r["query"]), parse_feats(r["selected_features(idx=weight)"])))

    if not rows:
        print(f"[err] seed {seed} 无数据")
        return 1
    print(f"[info] {species} seed {seed}: {len(rows)} 轮特征选择记录")

    # 按查询区间分桶：每桶内各特征平均权重（取该桶所有轮次的均值）
    buckets = defaultdict(list)  # bucket_start -> [ {feat: weight} ... ]
    max_q = rows[-1][0]
    for q, feats in rows:
        b = (q // bucket) * bucket
        buckets[b].append(feats)

    # 全局 top 特征（按全历程总权重），用于固定系列
    agg = defaultdict(float)
    for q, feats in rows:
        for k, w in feats.items():
            agg[k] += w
    top_names = [k for k, _ in sorted(agg.items(), key=lambda x: -x[1])[:8]]

    # 输出演化序列
    out_path = OUT / f"framework_species_feature_evolution_{species}_seed{seed}.csv"
    with io.open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_start", "query_end", "n_pos", "top1", "w_top1",
                    "top2", "w_top2", "top3", "w_top3",
                    "n_selected", "selected_share_top8", "feature_detail"])
        for b in sorted(buckets):
            recs = buckets[b]
            n_pos = 0
            agg_b = defaultdict(float)
            n_sel_set = set()
            for feats in recs:
                for k, v in feats.items():
                    agg_b[k] += v
                    n_sel_set.add(k)
            n_pos_est = recs[-1] if False else 0
            # 平均每轮特征数
            avg_n = sum(len(r) for r in recs) / len(recs)
            # top3 in bucket
            ranked = sorted(agg_b.items(), key=lambda x: -x[1] / len(recs))[:3]
            top1 = ranked[0][0] if ranked else ""
            w1 = round(ranked[0][1] / len(recs), 4) if ranked else 0.0
            top2 = ranked[1][0] if len(ranked) > 1 else ""
            w2 = round(ranked[1][1] / len(recs), 4) if len(ranked) > 1 else 0.0
            top3 = ranked[2][0] if len(ranked) > 2 else ""
            w3 = round(ranked[2][1] / len(recs), 4) if len(ranked) > 2 else 0.0
            # top8 覆盖份额（平均）
            share = 0.0
            for feats in recs:
                tot = sum(feats.values())
                if tot > 0:
                    share += sum(v for k, v in feats.items() if k in top_names) / tot
            share = round(share / len(recs), 3)
            detail = ";".join(f"{k}={v / len(recs):.3f}"
                              for k, v in sorted(agg_b.items(), key=lambda x: -x[1])[:8])
            w.writerow([b, min(b + bucket, max_q), n_pos, top1, w1, top2, w2,
                        top3, w3, round(avg_n, 1), share, detail])
    print(f"[done] 演化序列 -> {out_path}  （桶宽 {bucket} 查询，共 {len(buckets)} 桶）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
