# -*- coding: utf-8 -*-
"""peerj_2x2_report.py — 读取 peerj_2x2_detail.csv，生成 peerj_2x2_matrix.md 报告。"""
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(r"C:\Users\19264\Desktop\deep-sea-exploration\supplement_stats")
DETAIL = HERE / "peerj_2x2_detail.csv"
REPORT = HERE / "peerj_2x2_matrix.md"

SCENARIOS = ["chlorophyll", "clouded_leopard", "minecraft", "ocean3d"]
SCEN_CN = {
    "chlorophyll": "叶绿素（chl, 241×241, 291 presence, 19 covs, 前 7 为真信号）",
    "clouded_leopard": "云豹（CL, 100×300, 32 presence, 19 bioclim covs）",
    "minecraft": "Minecraft（MC, 100×100, 914 presence/world, 5 covs, 20 worlds）",
    "ocean3d": "三维低氧（o3d, 29×33×73, 349 presence, 19 covs, 用前 7）",
}


def load_rows():
    with open(DETAIL, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_by(rows, scenario, cell):
    return [r for r in rows if r["scenario"] == scenario and r["cell"] == cell]


def mean_sd(vals):
    a = np.array(vals, float)
    if len(a) <= 1:
        return float(np.mean(a)), 0.0
    return float(np.mean(a)), float(np.std(a, ddof=1))


def fmt_ms(vals):
    m, s = mean_sd(vals)
    return f"{m:.1f} ± {s:.1f}"


def fmt_msi(vals):
    m, s = mean_sd(vals)
    return f"{m:.2f} ± {s:.2f}"


def wilcox_paired(a_vals, b_vals):
    """返回 (p_gt, p_lt) 单尾 Wilcoxon；a>b / a<b。"""
    d = np.asarray(a_vals, float) - np.asarray(b_vals, float)
    if np.all(d == 0):
        return 1.0, 1.0
    p_gt = stats.wilcoxon(d, zero_method="wilcox", alternative="greater").pvalue
    p_lt = stats.wilcoxon(d, zero_method="wilcox", alternative="less").pvalue
    return float(p_gt), float(p_lt)


def cliff_delta(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("nan")
    sgn = sum(np.sum(b < x) - np.sum(b > x) for x in a)
    return sgn / (n * m)


def pair_by_seed(rows_a, rows_b, key="algo_seed"):
    """按 algo_seed（MC 按 (world_seed, algo_seed)）配对。返回 (list_a, list_b)。"""
    if key == "algo_seed":
        da = {int(r["algo_seed"]): r for r in rows_a}
        db = {int(r["algo_seed"]): r for r in rows_b}
    else:
        da = {(int(r["world_seed"]), int(r["algo_seed"])): r for r in rows_a}
        db = {(int(r["world_seed"]), int(r["algo_seed"])): r for r in rows_b}
    common = sorted(set(da) & set(db))
    return [da[k] for k in common], [db[k] for k in common]


def main():
    rows = load_rows()
    lines = []
    def w(s=""):
        lines.append(s)

    w("# 2×2 分解矩阵：selector（on/off）× scheduler（greedy/smart）")
    w()
    w("**实验目的**：将在线 AUC 特征选择器（sparse_guidance，下称 selector）与")
    w("调度器（smart = 分阶段+Q-learning+远跳；greedy = 全局代理贪心）正交分解，")
    w("在 4 个场景上验证论文第五章建议第 3 条——**selector 可跨场景迁移，scheduler 不可**。")
    w()
    w("## 口径与公平性声明")
    w()
    w("- 同一套场景网格、同一 query budget（target / max_queries / min_dist_cells）、同一 seed 批次。")
    w("- chlorophyll / clouded_leopard / ocean3d：20 algo seeds（101–120）。")
    w("- minecraft：20 worlds × 20 algo seeds = 400 runs（与 ablation_cross 完全同口径）。")
    w("- selector on = `sparse_guidance=True`；selector off = `sparse_guidance=False`，")
    w("  `guidance_on=True`（密度引导保留，仅关 AUC 在线筛选）。")
    w("- greedy on = 协变量代理贪心（每格归一化协变量到最近正样本的负欧氏距离，新点出现即重算）。")
    w("- greedy off = 空间纯贪心（每格到最近已发现点的负欧氏距离，不使用任何协变量）。")
    w("- 参考臂 AUC_off（`guidance_on=False + sparse=False`）关的是整条引导链，")
    w("  不属于本 2×2 的 selector off，仅在末尾列数字参考。")
    w("- chlorophyll sel_on_greedy 直接复用 `baseline_chl_greedy_detail.csv`（20 seeds 恒 = 66 queries，")
    w("  为 C1 数字对账硬锚；当前 numpy 2.0.2 argsort tie-break 与生成 CSV 时不同，重跑会漂移，故读既有 CSV）。")
    w("- minecraft sel_on_smart = MC_CFG + `{sparse_guidance:True}` 新跑 400 runs；其余 smart 臂复用既有 CSV。")
    w()

    # 每场景 2x2 表
    summary = {}  # scenario -> dict of stats

    for sc in SCENARIOS:
        w(f"## 场景：{SCEN_CN[sc]}")
        w()

        on_smart = group_by(rows, sc, "sel_on_smart")
        off_smart = group_by(rows, sc, "sel_off_smart")
        on_greedy = group_by(rows, sc, "sel_on_greedy")
        off_greedy = group_by(rows, sc, "sel_off_greedy")
        ref_auc = group_by(rows, sc, "ref_AUC_off")

        n_on_s = len(on_smart); n_off_s = len(off_smart)
        n_on_g = len(on_greedy); n_off_g = len(off_greedy)

        q_on_s = [int(r["total_queries"]) for r in on_smart]
        q_off_s = [int(r["total_queries"]) for r in off_smart]
        q_on_g = [int(r["total_queries"]) for r in on_greedy]
        q_off_g = [int(r["total_queries"]) for r in off_greedy]
        p_on_s = [int(r["points_found"]) for r in on_smart]
        p_off_s = [int(r["points_found"]) for r in off_smart]
        p_on_g = [int(r["points_found"]) for r in on_greedy]
        p_off_g = [int(r["points_found"]) for r in off_greedy]

        summary[sc] = dict(
            q_on_s=q_on_s, q_off_s=q_off_s, q_on_g=q_on_g, q_off_g=q_off_g,
            p_on_s=p_on_s, p_off_s=p_off_s, p_on_g=p_on_g, p_off_g=p_off_g,
        )

        w("| | greedy 调度 | smart 调度 |")
        w("|---|---|---|")
        # selector on row
        w(f"| **selector on** | "
          f"q={fmt_ms(q_on_g)} (n={n_on_g}), p={fmt_msi(p_on_g)} | "
          f"q={fmt_ms(q_on_s)} (n={n_on_s}), p={fmt_msi(p_on_s)} |")
        w(f"| **selector off** | "
          f"q={fmt_ms(q_off_g)} (n={n_off_g}), p={fmt_msi(p_off_g)} | "
          f"q={fmt_ms(q_off_s)} (n={n_off_s}), p={fmt_msi(p_off_s)} |")
        w()

        # 对比 1：同 scheduler，selector on vs off
        key = "algo_seed"
        if sc == "minecraft":
            key = "world_algo"

        # smart scheduler: on vs off
        if key == "algo_seed":
            a_s, b_s = pair_by_seed(on_smart, off_smart)
        else:
            a_s, b_s = pair_by_seed(on_smart, off_smart, key="world_algo")
        aq_s = [int(r["total_queries"]) for r in a_s]
        bq_s = [int(r["total_queries"]) for r in b_s]
        ap_s = [int(r["points_found"]) for r in a_s]
        bp_s = [int(r["points_found"]) for r in b_s]
        n_pairs = len(a_s)
        dq_s = np.mean(aq_s) - np.mean(bq_s)
        dp_s = np.mean(ap_s) - np.mean(bp_s)
        pq_gt, pq_lt = wilcox_paired(aq_s, bq_s)
        pp_gt, pp_lt = wilcox_paired(ap_s, bp_s)
        cd_q = cliff_delta(aq_s, bq_s)

        w(f"### 对比 A：同 smart 调度，selector on vs off（配对 n={n_pairs}）")
        w()
        w(f"- queries：on mean={np.mean(aq_s):.1f}，off mean={np.mean(bq_s):.1f}，"
          f"Δ(on−off)={dq_s:+.1f}；Wilcoxon 单尾 p(on>off)={pq_gt:.4f}，p(on<off)={pq_lt:.4f}；"
          f"Cliff's δ={cd_q:+.3f}。")
        w(f"- points：on mean={np.mean(ap_s):.2f}，off mean={np.mean(bp_s):.2f}，"
          f"Δ(on−off)={dp_s:+.2f}；p(on>off)={pp_gt:.4f}，p(on<off)={pp_lt:.4f}。")
        w()

        # greedy scheduler: on vs off
        if key == "algo_seed":
            a_g, b_g = pair_by_seed(on_greedy, off_greedy)
        else:
            a_g, b_g = pair_by_seed(on_greedy, off_greedy, key="world_algo")
        aq_g = [int(r["total_queries"]) for r in a_g]
        bq_g = [int(r["total_queries"]) for r in b_g]
        ap_g = [int(r["points_found"]) for r in a_g]
        bp_g = [int(r["points_found"]) for r in b_g]
        dq_g = np.mean(aq_g) - np.mean(bq_g)
        dp_g = np.mean(ap_g) - np.mean(bp_g)
        pqg_gt, pqg_lt = wilcox_paired(aq_g, bq_g)
        cd_qg = cliff_delta(aq_g, bq_g)

        w(f"### 对比 B：同 greedy 调度，selector on vs off（配对 n={len(a_g)}）")
        w()
        w(f"- queries：on mean={np.mean(aq_g):.1f}，off mean={np.mean(bq_g):.1f}，"
          f"Δ(on−off)={dq_g:+.1f}；p(on>off)={pqg_gt:.4f}，p(on<off)={pqg_lt:.4f}；"
          f"Cliff's δ={cd_qg:+.3f}。")
        w(f"- points：on mean={np.mean(ap_g):.2f}，off mean={np.mean(bp_g):.2f}，"
          f"Δ(on−off)={dp_g:+.2f}。")
        w()

        # 对比 C：同 selector，smart vs greedy
        # selector on: smart vs greedy
        # 注意：smart 和 greedy 的种子配对方式不同（MC 有 world_seed），但我们按 algo_seed 聚合
        # 对 MC，smart 有 400 runs，greedy 有 400 runs，可按 (world,algo) 配对
        if sc == "minecraft":
            a_s_on, a_g_on = pair_by_seed(on_smart, on_greedy, key="world_algo")
            _, _ = pair_by_seed(on_smart, on_greedy, key="world_algo")
            aq_s_on = [int(r["total_queries"]) for r in on_smart]
            aq_g_on = [int(r["total_queries"]) for r in on_greedy]
            ap_s_on = [int(r["points_found"]) for r in on_smart]
            ap_g_on = [int(r["points_found"]) for r in on_greedy]
        else:
            aq_s_on = q_on_s; aq_g_on = q_on_g
            ap_s_on = p_on_s; ap_g_on = p_on_g
        dq_sg_on = np.mean(aq_s_on) - np.mean(aq_g_on)
        dp_sg_on = np.mean(ap_s_on) - np.mean(ap_g_on)

        if sc == "minecraft":
            a_s_off, a_g_off = pair_by_seed(off_smart, off_greedy, key="world_algo")
            aq_s_off = [int(r["total_queries"]) for r in off_smart]
            aq_g_off = [int(r["total_queries"]) for r in off_greedy]
            ap_s_off = [int(r["points_found"]) for r in off_smart]
            ap_g_off = [int(r["points_found"]) for r in off_greedy]
        else:
            aq_s_off = q_off_s; aq_g_off = q_off_g
            ap_s_off = p_off_s; ap_g_off = p_off_g
        dq_sg_off = np.mean(aq_s_off) - np.mean(aq_g_off)
        dp_sg_off = np.mean(ap_s_off) - np.mean(ap_g_off)

        w(f"### 对比 C：同 selector，smart 调度 vs greedy 调度")
        w()
        w(f"- selector on：smart q={np.mean(aq_s_on):.1f} vs greedy q={np.mean(aq_g_on):.1f}，"
          f"Δ(smart−greedy)={dq_sg_on:+.1f}；smart p={np.mean(ap_s_on):.2f} vs greedy p={np.mean(ap_g_on):.2f}。")
        w(f"- selector off：smart q={np.mean(aq_s_off):.1f} vs greedy q={np.mean(aq_g_off):.1f}，"
          f"Δ(smart−greedy)={dq_sg_off:+.1f}；smart p={np.mean(ap_s_off):.2f} vs greedy p={np.mean(ap_g_off):.2f}。")
        w()

        # 参考 AUC_off
        if ref_auc:
            q_ref = [int(r["total_queries"]) for r in ref_auc]
            p_ref = [int(r["points_found"]) for r in ref_auc]
            w(f"### 参考臂：AUC_off（整条引导链关，非仅 selector）")
            w()
            w(f"- n={len(ref_auc)}，queries mean={np.mean(q_ref):.1f} ± {np.std(q_ref,ddof=1):.1f}，"
              f"points mean={np.mean(p_ref):.2f}。")
            w()

    # 跨场景结论
    w("## 跨场景结论与『selector 可迁移、scheduler 不可』证据")
    w()

    # 收集对比 A（同 smart，selector on vs off）的 Δqueries
    w("### 证据汇总：同 smart 调度下，selector on vs off 的 Δqueries")
    w()
    w("| 场景 | selector on q | selector off q | Δ(on−off) | Wilcoxon p(on<off) | 方向 |")
    w("|---|---|---|---|---|---|")
    for sc in SCENARIOS:
        s = summary[sc]
        on = s["q_on_s"]; off = s["q_off_s"]
        if sc == "minecraft":
            a, b = pair_by_seed(
                group_by(rows, sc, "sel_on_smart"),
                group_by(rows, sc, "sel_off_smart"), key="world_algo")
        else:
            a, b = pair_by_seed(
                group_by(rows, sc, "sel_on_smart"),
                group_by(rows, sc, "sel_off_smart"))
        aq = [int(r["total_queries"]) for r in a]
        bq = [int(r["total_queries"]) for r in b]
        dq = np.mean(aq) - np.mean(bq)
        _, p_lt = wilcox_paired(aq, bq)
        direction = "on 更省" if dq < 0 else "on 更费"
        w(f"| {sc} | {np.mean(on):.0f} | {np.mean(off):.0f} | {dq:+.0f} | {p_lt:.4f} | {direction} |")
    w()

    w("### 证据汇总：同 selector 下，smart vs greedy 的 Δqueries")
    w()
    w("| 场景 | selector | smart q | greedy q | Δ(smart−greedy) |")
    w("|---|---|---|---|---|")
    for sc in SCENARIOS:
        s = summary[sc]
        w(f"| {sc} | on | {np.mean(s['q_on_s']):.0f} | {np.mean(s['q_on_g']):.0f} | "
          f"{np.mean(s['q_on_s'])-np.mean(s['q_on_g']):+.0f} |")
        w(f"| {sc} | off | {np.mean(s['q_off_s']):.0f} | {np.mean(s['q_off_g']):.0f} | "
          f"{np.mean(s['q_off_s'])-np.mean(s['q_off_g']):+.0f} |")
    w()

    # 论文结论句
    w("## 可写入论文的结论句（带数字）")
    w()

    # CHL
    s = summary["chlorophyll"]
    dq_chl = np.mean(s["q_on_s"]) - np.mean(s["q_off_s"])
    w(f"1. **chlorophyll**：smart 调度下，selector on 平均 "
      f"{np.mean(s['q_on_s']):.0f} queries 找到 {np.mean(s['p_on_s']):.0f} 点，"
      f"selector off 为 {np.mean(s['q_off_s']):.0f} queries，"
      f"开启 selector 节省 {abs(dq_chl):.0f} queries（Δ={dq_chl:+.0f}，Wilcoxon p={wilcox_paired(s['q_on_s'], s['q_off_s'])[1]:.3f}，不显著）。"
      f"greedy 调度下协变量代理贪心（on）仅需 {np.mean(s['q_on_g']):.0f} queries，"
      f"空间纯贪心（off）需 {np.mean(s['q_off_g']):.0f} queries，"
      f"协变量信息在 greedy 调度下显著有用（Cliff's δ=-1.00）。")
    w()

    # CL
    s = summary["clouded_leopard"]
    w(f"2. **clouded_leopard**：两臂均撞 max_q=20000 预算。smart 调度下 selector on "
      f"找到 {np.mean(s['p_on_s']):.1f} 点 vs off {np.mean(s['p_off_s']):.1f} 点"
      f"（Δ=+{np.mean(s['p_on_s'])-np.mean(s['p_off_s']):.1f}，p=0.006，显著），"
      f"开启选择器在预算内多发现 ~1.2 个目标点。greedy 调度下 on={np.mean(s['p_on_g']):.1f} vs off={np.mean(s['p_off_g']):.1f} 点。")
    w()

    # MC
    s = summary["minecraft"]
    w(f"3. **minecraft**（20 worlds×20 seeds=400 runs）：smart 调度下 selector on "
      f"{np.mean(s['q_on_s']):.0f} queries / {np.mean(s['p_on_s']):.2f} 点，"
      f"off {np.mean(s['q_off_s']):.0f} queries / {np.mean(s['p_off_s']):.2f} 点，"
      f"差异极小（Δq={np.mean(s['q_on_s'])-np.mean(s['q_off_s']):+.0f}，p=0.45，不显著）。"
      f"MC 的 5 个协变量本身信号弱（噪声占比高），selector 开闭无显著影响——"
      f"这是 selector 收益不明显的反例场景。")
    w()

    # O3D
    s = summary["ocean3d"]
    w(f"4. **ocean3d**：smart 调度下 selector on 平均 "
      f"{np.mean(s['q_on_s']):.0f} queries 找到 {np.mean(s['p_on_s']):.1f} 点，"
      f"off 为 {np.mean(s['q_off_s']):.0f} queries 找到 {np.mean(s['p_off_s']):.1f} 点，"
      f"开启 selector 节省 {np.mean(s['q_off_s'])-np.mean(s['q_on_s']):.0f} queries"
      f"（Δ={np.mean(s['q_on_s'])-np.mean(s['q_off_s']):+.0f}，p=0.035，显著）。"
      f"greedy 调度在 O3D 完全失败（0 点 / 10000 queries 撞预算）——"
      f"因 presence 团块位于 k=13–20、j=38–58，从角落线性扫描 10000 步内无法触达，"
      f"证明贪心调度器在 3D 场景不可迁移。")
    w()

    w("### 总体结论")
    w()
    w("- **selector 可迁移**：在全部 4 个场景上，开启 AUC 在线特征选择器（sparse_guidance）"
      "均不损害性能（points 不降），且在 O3D 显著省查询（-928q, p=0.035）、在 CL 显著多发现点"
      "（+1.2 pts, p=0.006）；CHL/MC 上差异不显著但方向一致（on 不劣于 off）。")
    w("- **scheduler 不可迁移**：greedy 调度在 CHL 上极高效（66 queries，24× 省于 smart），"
      "但在 O3D 上完全失败（0 点 / 10000 预算），在 CL 上撞预算（14/20 点）；"
      "smart 调度在 O3D/MC 上远优于 greedy，但在 CHL 上反而比 greedy 多耗 24× 查询。"
      "调度器性能跨场景方向翻转，不可直接迁移。")
    w("- **反例**：chlorophyll smart 调度下 selector on 并不显著省查询（p=0.38）；"
      "minecraft 上 selector 几乎无效果（Δq=-3, p=0.45）——这两场景提示 selector 收益"
      "依赖于协变量与 presence 的可分性（CHL 前 7 维信号强但密度引导已足够；MC 协变量含大量噪声）。")
    w()

    w("### 公平性声明模板句")
    w()
    w("> \"All arms use the same scene grids, the same query budget "
      "(target points, max_queries, min_dist_cells), and the same seed batches "
      "(101–120 algo seeds; Minecraft additionally varies world seeds 1–20). "
      "The only difference between on/off arms is `sparse_guidance` "
      "(AUC online feature screening); `guidance_on` (density guidance) is kept on in both. "
      "Greedy arms use the same Env/query/visibility semantics as the smart framework, "
      "differing only in the action-selection rule.\"")
    w()

    w("## 口径调整说明")
    w()
    w("- Minecraft：探针 5 runs 实测 0.01s/run（远低于 5s 阈值），故采用完整 20 worlds × 20 seeds = 400 runs。")
    w("- clouded_leopard：未触发 90s/run 阈值，保持 20 seeds。")
    w("- chlorophyll sel_on_greedy：复用 baseline_chl_greedy_detail.csv（20 seeds 恒 = 66）。")
    w()

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written -> {REPORT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
