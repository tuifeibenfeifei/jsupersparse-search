# PeerJ Revision Supplement (Experiments Added for PeerJ Computer Science Resubmission)

All numbers below are recomputable from the CSVs in this directory (`peerj_*.csv`) with the
corresponding `peerj_*.py` scripts, using the framework in `src/`. Every added experiment
reuses the scene oracles, grids, budgets, and seed batches of the paper; only the listed
mechanism flags differ between arms. Paired tests are one-sided Wilcoxon signed-rank
(`scipy.stats.wilcoxon`, `zero_method='wilcox'`, α = 0.05) unless stated; effect sizes are
Cliff's delta. `n.s.` denotes not significant; a non-significant contrast is reported as a
direction in the data, not an established effect.

## 1. Unified statistics and confidence intervals (`peerj_stats_unified.py`, `peerj_stats_unified.csv`)

Recomputes the paper's paired comparisons from existing CSVs (no experiments rerun):

- Chlorophyll: AUC screen off costs +209.05 queries (p = 1.9e-6, Cliff's delta = +0.900);
  multicluster off saves −637.80 queries (p = 9.5e-7, delta = −1.000). Both saturate at
  20/20 points in every seed.
- Clouded leopard: multicluster off loses −1.85 points (p = 0.011, delta = −0.600);
  AUC off is n.s. (direction only).
- Minecraft (400 paired runs): all contrasts n.s. (worst p = 0.063, direction only).
- Selection precision 0.763 vs chance 0.368: a pooled Wilson 95% CI computed from the
  controlled screen log (`screen_mi_vs_auc.csv`) gives precision 0.973 [0.922, 0.991]
  (106 true of 109 retained); pooling the main-line 0.763 over n in {109, 218, 300, 436}
  gives bounds 0.675–0.833. In every case chance 0.368 lies far below the interval.
- Anchor check: chlorophyll full / AUC-off / multicluster-off 20-seed mean queries =
  1596.25 / 1805.30 / 958.45, matching `ablation_detail.csv` within ≤ 5.

## 2. One-at-a-time mechanism × scene ablation matrix (`peerj_oat_ablation.py`, `peerj_oat_detail.csv`, `peerj_oat_matrix.md`)

Each cell disables one mechanism (OAT) vs full, paired per algo seed (Minecraft also per
world seed), 2322 runs in `peerj_oat_detail.csv`. Seventeen mechanism-by-scene cells:

- Chlorophyll (n = 20): AUC density guidance off +200 q (p < 0.001) → beneficial;
  multicluster off −647 q (p < 0.001) → harmful; blank_jump off −650 q (p < 0.001) → harmful;
  serpentine stage-1 scan +1384 q and −9.10 points (p < 0.001 / p = 0.003) → spread sweep beneficial;
  directed_jump off +1.4 q (p = 0.11) → inert; n_bins 4 / 16 → inert / weakly beneficial (p = 0.030).
- Clouded leopard (n = 20; all arms saturate max_queries = 20000, so comparisons are on points):
  multicluster off −1.85 points (p = 0.022) → beneficial; directed_jump off −1.10 (p = 0.042) → beneficial;
  blank_jump off −1.25 (p = 0.051, marginal); AUC off n.s.; n_bins inert.
- Minecraft (n = 400): follow_mode off +81 q (p < 0.001) → beneficial; AUC / multicluster /
  blank_jump off n.s. → inert; directed_jump is off in the baseline config (not tested).
- Follow mode and serpentine are only exercised where the baseline enables them; both
  two-seed / twenty-seed re-runs confirm the "already off in baseline" cases bit-identically.

## 3. Selector × scheduler 2×2 decomposition (`peerj_2x2_matrix.py`, `peerj_2x2_detail.csv`)

Crosses the online AUC feature selector (on/off, `sparse_guidance`) with the scheduler
(smart staged framework vs a greedy action rule) on four scenes, same grids/budgets/seeds:

- Chlorophyll: greedy+selector 66 q; greedy only 74 q (Cliff's delta = −1.000); smart+selector
  1596 q vs smart only 1617 q (Δ = −21, p = 0.378, n.s.).
- Clouded leopard: smart+selector 14.00 points vs smart only 12.80 (Δ = +1.20, p = 0.006).
- Minecraft (400 runs): smart+selector 3899.5 q / 29.75 points vs 3902.6 q / 29.80 (n.s.;
  covariates carry little signal here — a counter-example scene).
- Ocean3D: smart+selector 3038.9 q / 39.30 points vs 3966.8 q / 40.00 (Δ = −928, p = 0.035);
  greedy fails outright (0 points in 10000 queries) because the presence blocks lie out of a
  linear sweep reach — the scheduler does not transfer to 3-D.
- Conclusion supported: the selector never hurts and transfers across scenes; the scheduler's
  ranking flips across scenes and does not transfer.

## 4. End-to-end wall-clock vs GP-UCB / TuRBO (`peerj_wallclock.py`, `peerj_wallclock_detail.csv`)

Same machine, same oracle, serial runs, `time.perf_counter`, no GPU:

- Chlorophyll: GP-UCB 93.5 queries / 3.70 s total wall-clock (about 10x faster than the
  framework's 1605.2 q / 37.27 s); TuRBO 329.4 q but 79.97 s (2.1x the framework) because
  GPR refitting grows with history.
- Minecraft (worlds 1–10): framework 2587.4 q / 66.16 s vs GP-UCB 695.8 q / 1300.82 s
  (19.7x) and TuRBO 499.4 q / 1600.74 s (24.2x); 46/52 GP runs cut by a 20-s watchdog,
  so those totals are lower bounds.
- Micro-benchmark: GP single-step cost grows from 6.52 ms at n = 100 to 2951.57 ms at
  n = 4000 (453x for 40x more samples); the framework stays at 1.53 ms per query,
  independent of observation count. The framework trades more queries for a constant
  per-query cost; the comparison is reported without a pre-committed direction.

## 5. n_bins sensitivity and hyperparameter transparency (`peerj_nbins_scan.py`, `peerj_nbins_scan.csv`)

Chlorophyll, seeds 101–105, all arms reach 20/20 points:

- n_bins = 4: 1393.0 q ｜ n_bins = 8 (default): 1267.2 q ｜ n_bins = 16: 1377.6 q.
  The default 8 is near-optimal and robust.

## 6. Data provenance verification (`peerj_data_provenance.md`)

Scenario-by-scenario audit (2026-10-16) of target-field data sources, presence-field
construction, and covariate sources, checked against the construction scripts
(`species-search/build_oracle.py`, `species-search/fetch_occurrences.ps1`,
`chlorophyll-search/fetch_chlorophyll.py`, `ocean3d-search/fetch_ocean3d.py`).
Raw third-party data are not distributed in this repository (see README "Data acquisition").

## 7. Held-out seeds and selection-purity growth (`heldout_seeds.py`, `peerj_proxy_only.py`)

- Held-out seed block 201–220 (chlorophyll): full framework 1706 vs 1045 queries
  (p < 0.0001, delta 0.72); 15.30 vs 13.35 points (p = 0.0054).
- Selection purity rises with warm-up: the proxy screen's retained-signal fraction
  increases as samples accumulate (`peerj_proxy_growth.csv`).

## Files in this directory (PeerJ-added)

| File | Content |
|---|---|
| `peerj_stats_unified.py/.csv` | Unified paired comparisons + Wilson CIs |
| `peerj_oat_ablation.py`, `peerj_oat_detail.csv`, `peerj_oat_post.py` | One-at-a-time mechanism ablation (2322 runs) |
| `peerj_2x2_matrix.py`, `peerj_2x2_detail.csv` | Selector × scheduler decomposition (4 scenes) |
| `peerj_wallclock.py`, `peerj_wallclock_detail.csv` | Wall-clock vs GP-UCB / TuRBO + micro-benchmark |
| `peerj_nbins_scan.py/.csv` | n_bins ∈ {4, 8, 16} sensitivity |
| `peerj_proxy_only.py`, `peerj_proxy_growth.csv` | Selection-purity growth with warm-up |
| `peerj_resume_mc.py` | Minecraft run resumption (crash-safe driver) |
| `peerj_hash_test.py/.txt`, `peerj_mc_anchor_check.py/.txt`, `peerj_mc_presence_stats.py` | Anchor/consistency checks for MC numbers |

All statistics in the PeerJ revision of the manuscript (Tables 9–17 and the Discussion)
derive from these files plus the pre-existing `results/` CSVs.
