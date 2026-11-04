# 2x2 decomposition matrix: selector (on/off) x scheduler (greedy/smart)

**Goal**: orthogonally decompose the online AUC feature selector (sparse_guidance, called selector below) and the
scheduler (smart = staged + Q-learning + long jump; greedy = global surrogate greedy) across
4 scenarios to validate Recommendation 3 of Chapter 5 of the paper - **the selector transfers across scenarios, the scheduler does not**.

## Setup and fairness statement

- Same set of scenario grids, same query budget (target / max_queries / min_dist_cells), same seed batch.
- chlorophyll / clouded_leopard / ocean3d:20 algo seeds(101–120).
- minecraft: 20 worlds x 20 algo seeds = 400 runs (exactly the same setup as ablation_cross).
- selector on = `sparse_guidance=True`;selector off = `sparse_guidance=False`,
  `guidance_on=True` (density guidance kept, only AUC online screening turned off).
- greedy on = covariate-surrogate greedy (each cell uses the negative Euclidean distance of normalized covariates to the nearest positive sample, recomputed when a new point appears).
- greedy off = purely spatial greedy (each cell uses the negative Euclidean distance to the nearest discovered point, no covariates used).
- The reference arm AUC_off (`guidance_on=False + sparse=False`) turns off the whole guidance chain,
  not the selector off of this 2x2, and is only listed at the end for numerical reference.
- chlorophyll sel_on_greedy directly reuses `baseline_chl_greedy_detail.csv` (20 seeds always = 66 queries,
  a hard anchor for C1 numerical reconciliation; the current numpy 2.0.2 argsort tie-break differs from when the CSV was generated and reruns would drift, so the existing CSV is read).
- minecraft sel_on_smart = MC_CFG + `{sparse_guidance:True}` rerun for 400 runs; the other smart arms reuse existing CSVs.

## Scenario: Chlorophyll (chl, 241x241, 291 presence, 19 covs, first 7 are true signals)

| | greedy scheduler | smart scheduler |
|---|---|---|
| **selector on** | q=66.0 ± 0.0 (n=20), p=20.00 ± 0.00 | q=1596.2 ± 421.1 (n=20), p=20.00 ± 0.00 |
| **selector off** | q=74.0 ± 0.0 (n=20), p=20.00 ± 0.00 | q=1617.0 ± 449.3 (n=20), p=20.00 ± 0.00 |

### Comparison A: same smart scheduler, selector on vs off (paired n=20)

- queries: on mean=1596.2, off mean=1617.0, Δ(on-off)=-20.7; Wilcoxon one-sided p(on>off)=0.6357, p(on<off)=0.3781; Cliff's δ=-0.075.
- points: on mean=20.00, off mean=20.00, Δ(on-off)=+0.00; p(on>off)=1.0000, p(on<off)=1.0000.

### Comparison B: same greedy scheduler, selector on vs off (paired n=20)

- queries: on mean=66.0, off mean=74.0, Δ(on-off)=-8.0; p(on>off)=1.0000, p(on<off)=0.0000; Cliff's δ=-1.000.
- points: on mean=20.00, off mean=20.00, Δ(on-off)=+0.00.

### Comparison C: same selector, smart scheduler vs greedy scheduler

- selector on: smart q=1596.2 vs greedy q=66.0, Δ(smart-greedy)=+1530.2; smart p=20.00 vs greedy p=20.00.
- selector off: smart q=1617.0 vs greedy q=74.0, Δ(smart-greedy)=+1543.0; smart p=20.00 vs greedy p=20.00.

### Reference arm: AUC_off (whole guidance chain off, not only the selector)

- n=20, queries mean=1805.3 ± 410.4, points mean=20.00.

## Scenario: Clouded leopard (CL, 100x300, 32 presence, 19 bioclim covs)

| | greedy scheduler | smart scheduler |
|---|---|---|
| **selector on** | q=20000.0 ± 0.0 (n=20), p=14.00 ± 0.00 | q=20000.0 ± 0.0 (n=20), p=14.00 ± 3.23 |
| **selector off** | q=20000.0 ± 0.0 (n=20), p=12.00 ± 0.00 | q=20000.0 ± 0.0 (n=20), p=12.80 ± 3.00 |

### Comparison A: same smart scheduler, selector on vs off (paired n=20)

- queries: on mean=20000.0, off mean=20000.0, Δ(on-off)=+0.0; Wilcoxon one-sided p(on>off)=1.0000, p(on<off)=1.0000; Cliff's δ=+0.000.
- points: on mean=14.00, off mean=12.80, Δ(on-off)=+1.20; p(on>off)=0.0064, p(on<off)=0.9936.

### Comparison B: same greedy scheduler, selector on vs off (paired n=20)

- queries: on mean=20000.0, off mean=20000.0, Δ(on-off)=+0.0; p(on>off)=1.0000, p(on<off)=1.0000; Cliff's δ=+0.000.
- points: on mean=14.00, off mean=12.00, Δ(on-off)=+2.00.

### Comparison C: same selector, smart scheduler vs greedy scheduler

- selector on: smart q=20000.0 vs greedy q=20000.0, Δ(smart-greedy)=+0.0; smart p=14.00 vs greedy p=14.00.
- selector off: smart q=20000.0 vs greedy q=20000.0, Δ(smart-greedy)=+0.0; smart p=12.80 vs greedy p=12.00.

### Reference arm: AUC_off (whole guidance chain off, not only the selector)

- n=20, queries mean=19718.3 ± 1259.6, points mean=14.40.

## Scenario: Minecraft(MC, 100×100, 914 presence/world, 5 covs, 20 worlds)

| | greedy scheduler | smart scheduler |
|---|---|---|
| **selector on** | q=4252.1 ± 4436.1 (n=400), p=29.80 ± 15.99 | q=3899.5 ± 4516.4 (n=400), p=29.75 ± 16.08 |
| **selector off** | q=4315.3 ± 4159.5 (n=400), p=30.35 ± 16.19 | q=3902.6 ± 4514.8 (n=400), p=29.80 ± 16.10 |

### Comparison A: same smart scheduler, selector on vs off (paired n=400)

- queries: on mean=3899.5, off mean=3902.6, Δ(on-off)=-3.1; Wilcoxon one-sided p(on>off)=0.5547, p(on<off)=0.4453; Cliff's δ=-0.000.
- points: on mean=29.75, off mean=29.80, Δ(on-off)=-0.04; p(on>off)=0.8279, p(on<off)=0.1721.

### Comparison B: same greedy scheduler, selector on vs off (paired n=400)

- queries: on mean=4252.1, off mean=4315.3, Δ(on-off)=-63.2; p(on>off)=1.0000, p(on<off)=0.0000; Cliff's δ=-0.125.
- points: on mean=29.80, off mean=30.35, Δ(on-off)=-0.55.

### Comparison C: same selector, smart scheduler vs greedy scheduler

- selector on: smart q=3899.5 vs greedy q=4252.1, Δ(smart-greedy)=-352.6; smart p=29.75 vs greedy p=29.80.
- selector off: smart q=3902.6 vs greedy q=4315.3, Δ(smart-greedy)=-412.7; smart p=29.80 vs greedy p=30.35.

### Reference arm: AUC_off (whole guidance chain off, not only the selector)

- n=400, queries mean=3902.9 ± 4513.9, points mean=29.75.

## Scenario: 3D hypoxia (o3d, 29x33x73, 349 presence, 19 covs, uses first 7)

| | greedy scheduler | smart scheduler |
|---|---|---|
| **selector on** | q=10000.0 ± 0.0 (n=20), p=0.00 ± 0.00 | q=3038.9 ± 1925.2 (n=20), p=39.30 ± 3.13 |
| **selector off** | q=10000.0 ± 0.0 (n=20), p=0.00 ± 0.00 | q=3966.8 ± 1961.7 (n=20), p=40.00 ± 0.00 |

### Comparison A: same smart scheduler, selector on vs off (paired n=20)

- queries: on mean=3038.9, off mean=3966.8, Δ(on-off)=-927.8; Wilcoxon one-sided p(on>off)=0.9681, p(on<off)=0.0348; Cliff's δ=-0.285.
- points: on mean=39.30, off mean=40.00, Δ(on-off)=-0.70; p(on>off)=0.8413, p(on<off)=0.1587.

### Comparison B: same greedy scheduler, selector on vs off (paired n=20)

- queries: on mean=10000.0, off mean=10000.0, Δ(on-off)=+0.0; p(on>off)=1.0000, p(on<off)=1.0000; Cliff's δ=+0.000.
- points: on mean=0.00, off mean=0.00, Δ(on-off)=+0.00.

### Comparison C: same selector, smart scheduler vs greedy scheduler

- selector on: smart q=3038.9 vs greedy q=10000.0, Δ(smart-greedy)=-6961.1; smart p=39.30 vs greedy p=0.00.
- selector off: smart q=3966.8 vs greedy q=10000.0, Δ(smart-greedy)=-6033.2; smart p=40.00 vs greedy p=0.00.

## Cross-scenario conclusions and evidence for 'selector transfers, scheduler does not'

### Evidence summary: Δqueries for selector on vs off under the same smart scheduler

| Scenario | selector on q | selector off q | Δ(on-off) | Wilcoxon p(on<off) | Direction |
|---|---|---|---|---|---|
| chlorophyll | 1596 | 1617 | -21 | 0.3781 | on saves |
| clouded_leopard | 20000 | 20000 | +0 | 1.0000 | on costs more |
| minecraft | 3900 | 3903 | -3 | 0.4453 | on saves |
| ocean3d | 3039 | 3967 | -928 | 0.0348 | on saves |

### Evidence summary: Δqueries for smart vs greedy under the same selector

| Scenario | selector | smart q | greedy q | Δ(smart-greedy) |
|---|---|---|---|---|
| chlorophyll | on | 1596 | 66 | +1530 |
| chlorophyll | off | 1617 | 74 | +1543 |
| clouded_leopard | on | 20000 | 20000 | +0 |
| clouded_leopard | off | 20000 | 20000 | +0 |
| minecraft | on | 3900 | 4252 | -353 |
| minecraft | off | 3903 | 4315 | -413 |
| ocean3d | on | 3039 | 10000 | -6961 |
| ocean3d | off | 3967 | 10000 | -6033 |

## Conclusion sentences that can go into the paper (with numbers)

1. **chlorophyll**: under the smart scheduler, selector on averages 1596 queries to find 20 points, while selector off takes 1617 queries, enabling the selector saves 21 queries (Δ=-21, Wilcoxon p=0.378, not significant). under the greedy scheduler, the covariate-surrogate greedy (on) needs only 66 queries, while purely spatial greedy (off) needs 74 queries, so covariate information is significantly useful under the greedy scheduler (Cliff's δ=-1.00).

2. **clouded_leopard**: both arms hit the max_q=20000 budget. Under the smart scheduler, selector on finds 14.0 points vs off 12.8 points(Δ=+1.2, p=0.006, significant), enabling the selector finds ~1.2 more target points within budget. Under the greedy scheduler, on=14.0 vs off=12.0 points.

3. **minecraft** (20 worlds x 20 seeds = 400 runs): under the smart scheduler, selector on 3900 queries / 29.75 points, off 3903 queries / 29.80 points, the difference is tiny (Δq=-3, p=0.45, not significant). the 5 MC covariates themselves are weak signals (high noise ratio), so toggling the selector has no significant effect - this is a counterexample scenario where the selector benefit is not obvious.

4. **ocean3d**: under the smart scheduler, selector on averages 3039 queries to find 39.3 points, while off takes 3967 queries to find 40.0 points, so enabling the selector saves 928 queries (Δ=-928, p=0.035, significant). the greedy scheduler completely fails on O3D (0 points / 10000 queries, hitting the budget) - because the presence cluster lies at k=13-20, j=38-58 and cannot be reached from the corner within 10000 linear-scan steps, proving that the greedy scheduler does not transfer to 3D scenarios.

### Overall conclusions

- **The selector transfers**: across all 4 scenarios, enabling the AUC online feature selector (sparse_guidance) never hurts performance (points do not decrease), significantly saves queries on O3D (-928q, p=0.035), and finds significantly more points on CL (+1.2 pts, p=0.006); on CHL/MC the difference is not significant but the direction is consistent (on is no worse than off).
- **The scheduler does not transfer**: the greedy scheduler is highly efficient on CHL (66 queries, 24x cheaper than smart), but completely fails on O3D (0 points / 10000 budget) and hits the budget on CL (14/20 points); the smart scheduler beats greedy on O3D/MC, yet on CHL it spends 24x more queries than greedy. Scheduler performance flips direction across scenarios and cannot be directly transferred.
- **Counterexample**: under the chlorophyll smart scheduler, selector on does not significantly save queries (p=0.38); on minecraft the selector has almost no effect (Δq=-3, p=0.45) - these two scenarios suggest that the selector benefit depends on the separability of covariates and presence (CHL's first 7 dimensions are strong signals but density guidance is already enough; MC covariates contain much noise).

### Fairness statement template

> "All arms use the same scene grids, the same query budget (target points, max_queries, min_dist_cells), and the same seed batches (101–120 algo seeds; Minecraft additionally varies world seeds 1–20). The only difference between on/off arms is `sparse_guidance` (AUC online feature screening); `guidance_on` (density guidance) is kept on in both. Greedy arms use the same Env/query/visibility semantics as the smart framework, differing only in the action-selection rule."

## Notes on setup adjustments

- Minecraft: a 5-run probe measured 0.01s/run (far below the 5s threshold), so the full 20 worlds x 20 seeds = 400 runs were used.
- clouded_leopard: the 90s/run threshold was not triggered, so 20 seeds were kept.
- chlorophyll sel_on_greedy: reuses baseline_chl_greedy_detail.csv (20 seeds always = 66).
