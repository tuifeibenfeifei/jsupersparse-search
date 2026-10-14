# jsupersparse-search

Source code for **"Density-Binned Online Covariate Selection for Black-Box Search of Sparse Targets"** (IEEE Access, under review).

A model-free, multistage black-box online search framework that finds sparse targets over spatial grids under a fixed query budget:

- **Stage 1** – cold-start coverage sweep that spreads an unbiased fraction of the budget;
- **Stage 2** – variance-prefiltered, budgeted AUC screening that selects the informative covariates online;
- **Stage 3** – tabular Q-learning over coarse block coordinates with directed far jumps, reward signals, and curiosity exploration.

The framework is evaluated on **eight scenarios**: Minecraft terrain, Tufted deer (Muntiacus crinifrons), Clouded leopard (Neofelis nebulosa), chlorophyll fields, 3D oxygen-poor nuclei, slow drift, mutation, and maritime search-and-rescue.

## Repository structure

```
src/                 Framework implementation (main algorithm, state manager, baselines, chunk finders, reward modules, traversal strategies)
scripts/             Experiment drivers, ablation scripts, and figure generation
configs/             YAML configuration files (final model and ablation arms)
minecraft-search/    Minecraft oracle construction and MC experiments
species-search/      Tufted deer / Clouded leopard oracles and experiments
chlorophyll-search/  Chlorophyll oracle (ESA CCI) and experiments
ocean3d-search/      3D oxygen-nucleus oracle (EMODnet) and experiments
drift-search/        Slow-drift and mutation oracles and experiments
maritime-search/     Maritime search-and-rescue oracle and experiments
results/             Per-seed and summary CSV outputs behind the paper's tables and figures
supplement_stats/    Paired tests, sensitivity sweeps, permutation controls, seed-stability and metric-robustness analyses
```

## Requirements

- Python 3.9+ (`pip install -r requirements.txt`)
- **Minecraft scenario only**: [cubiomes](https://github.com/Cubitect/cubiomes), a C reimplementation of the Minecraft terrain generator (check its license). Build it yourself and place the shared library as `src/cubiomes.dll`; the wrapper is `src/cubiomes_wrapper.py`. All other scenarios are pure Python.

## Data acquisition

Raw third-party data are **not distributed** in this repository; download them as follows before running the corresponding scenarios:

| Scenario | Data | Source |
|---|---|---|
| Tufted deer / Clouded leopard | GBIF occurrence records + WorldClim bioclimatic variables | GBIF (https://www.gbif.org) and WorldClim (https://www.worldclim.org); run `species-search/fetch_occurrences.ps1` |
| Chlorophyll | ESA Ocean Colour CCI chlorophyll-a product | https://esa-oceancolour-cci.org; run `chlorophyll-search/fetch_chlorophyll.py` |
| 3D oxygen | EMODnet "Dissolved oxygen and temperature on water column, data from the EGIM, EMSO-Canarias (ESTOC), 2020" | doi:10.17882/83639 (CC-BY 4.0); run `ocean3d-search/fetch_ocean3d.py` |
| Slow drift / Mutation / Maritime SAR | Synthetic, code-generated | no external data required |

## Reproduction

Run each experiment from the repository root:

- **Minecraft** – `minecraft-search/run_mc_framework.py` (framework), `run_mc_baselines.py`, `run_mc_gpucb.py` (baselines), `run_mc_budget_curve.py` (budget sweep)
- **Species** – `species-search/run_framework_sparse.py` (Tufted deer / Clouded leopard)
- **Chlorophyll** – `chlorophyll-search/run_chl.py`
- **3D oxygen** – `ocean3d-search/run_ocean3d.py`
- **Slow drift / Mutation** – `drift-search/run_drift.py`
- **Maritime** – `maritime-search/run_maritime.py`
- **Baselines (all scenarios)** – `scripts/run_baselines.py`, `scripts/run_finder_showdown.py`
- **Ablations** – `scripts/run_ablation_stage1.py` … `run_ablation_stage4.py`, `scripts/run_stages_parallel.py`
- **Final configuration** – `scripts/run_final_model.py`
- **Supplement statistics** – `supplement_stats/*.py` (paired tests, mutation sweep, permutation control, seed stability, metric robustness, held-out analysis)
- **Figures** – `scripts/generate_fig1.py`, `scripts/generate_fig8.py`, `scripts/generate_graphical_abstract.py`, plus the per-scenario `make_*_plot.py` scripts

Result CSVs are written under `results/` and `supplement_stats/`, matching the tables and figures in the paper.

## License

MIT — see [LICENSE](LICENSE).

## Citation

TBD once the article is accepted (IEEE Access, 2026).
