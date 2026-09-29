# `scripts/`

Command-line entry points. Run all of them from the repository root with the
`GRTL` conda environment active.

## Running experiments

| Script | What it does |
|---|---|
| `run_revision_queue.py` | the batch runner for the current revision. Reads `docs/revision/REVISION_EXECUTION_ORDER.md`, runs each unchecked config in its own subprocess, ticks the box when the results file appears, and resumes wherever it stopped. `--workers N`, `--gpu`. |
| `run_experiments.py` | runs a fixed dataset x generator x minimizer matrix to completion, fold by fold. `run_revision_queue.py` spawns this one per config. |
| `gen_revision_configs.py` | regenerates the `lab/config/generate_minimize/` tree from the protocol in `docs/revision/REVISION_EXPERIMENTS.md`. |

## Pre-training the shared artefacts

Both write a small pickle keyed by the dataset hash under
`lab/data/cache/explainers/`, which every later run picks up automatically.
Run them once per dataset before the configs that need them.

| Script | Artefact |
|---|---|
| `compute_dcm.py` | `DCM-*`, the distance-based class medoids |
| `compute_lst_methods.py` | `LSTMethodsArtifact-*`, the node-attribute method scores shared by every trainable LocalSearch variant |

## Figures

| Script | Output |
|---|---|
| `make_paper_figures.py` | the per-generator 2x2 result figures for the revision |
| `plot_final_results.py` | the per-metric grouped bars for the thesis results section |
| `_results_agg.py` | not an entry point: the aggregation both figure scripts and `lab/notebooks/stats_visualizer.ipynb` share |

## Checks

The smoke tests live in `tests/`: `regression_smoke.py` for the current matrix,
`catalogue_smoke.py` for the published baselines (MEG, MACCS, pRand, DDBS,
CounteRGAN, EAGER, RSGG), which no current experiment exercises. The repository
integrity checks live in `tools/`.

## Environment

`setup-grtl-gpu.sh` builds the `GRTL` conda environment with CUDA wheels.
