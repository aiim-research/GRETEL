# `lab/` - the experiment workbench

Everything that belongs to running experiments rather than to the framework
itself. `src/` is the library; this is where it gets driven.

```
lab/
├── config/          experiment configurations (see below)
├── notebooks/       analysis notebooks, current ones at the top level
│   └── legacy/      the retired generation
├── data/cache/      datasets, oracles and explainer artefacts, keyed by hash
├── graphics/        figures and the CSVs behind them
└── output/          results and logs (not versioned)
```

## `lab/config/`

| Tree | What it drives |
|---|---|
| `generate_minimize/` | the main matrix: dataset x generator x minimizer, one directory per combination, ten folds each. This is what `docs/revision/REVISION_EXECUTION_ORDER.md` queues. |
| `tagging/` | the tagging-strategy study (one subtree per tagger) |
| `bls_selection_net/`, `bls_selection_net_trainable/` | the learned edge-selector variants of local search |
| `meta_ens/`, `ensembles/` | ensemble aggregation and explainer selection |
| `metaheuristics/`, `meta/`, `llm_exp_generate_minimize/`, `debug/` | smaller studies |
| `snippets/` | the shared pieces every config composes: `do-pairs/` (dataset + oracle), `datasets/` (manipulator sets), pipelines, store paths |
| `base/` | minimal single-explainer examples, a good place to start reading |
| `legacy/` | retired one-off configs from earlier phases |

A config is assembled by the composer: `compose_do`, `compose_man`,
`compose_pip` and `compose_strs` are replaced by the contents of the snippet
they name **before** anything is hashed, so snippet paths can move freely.
Class paths cannot - see `tools/README.md`.

## `lab/data/cache/`

Directory names are content hashes (`ASD-15273954d84e...`), computed from the
component's resolved configuration. Datasets and oracles are rebuilt on demand
and stay out of git. Two kinds of artefact are small and expensive enough to
be versioned, under `cache/explainers/<dataset-hash>/`:

* `DCM-*` - the distance-based class medoids, trained by `scripts/compute_dcm.py`
* `LSTMethodsArtifact-*` - the shared node-attribute method scores, trained by
  `scripts/compute_lst_methods.py`

## `lab/notebooks/`

Current: `stats_visualizer.ipynb` (the revision's tables and figures),
`stats_visualizer_global.ipynb`, `testing_pipeline.ipynb`,
`1-evaluation_pipeline.ipynb`, `1-evaluation_pipeline_llm_explanation.ipynb`,
`subex_pipeline.ipynb`.

Each starts by chdir-ing to the repository root, so run them from anywhere but
expect paths inside to be root-relative.
