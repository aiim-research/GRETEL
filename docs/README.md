# Documentation map

## Start here

* [Reproducing experiments](reproducing-experiments.md) walks from a fresh clone to a reproduced number: environment, data, how a config is assembled, how to run one experiment or the whole batch, and where results land.
* [Thesis experiments](thesis-experiments.md) maps every experiment in the LBS thesis to the configuration directory that produces it, including the name translation (BLS is `lcls`, OBS-min is `obs`, DCEM is `dcm`).
* The top-level [README](../README.md) covers what GRETEL is, what ships with it, and how to cite it.

## Where things live

| Directory | Contents |
|---|---|
| [`src/`](../src) | the framework: datasets, oracles, embedders, explainers, evaluation |
| [`lab/`](../lab) | the experiment workbench: configurations, notebooks, caches, results. See [`lab/README.md`](../lab/README.md) |
| [`scripts/`](../scripts) | command-line entry points. See [`scripts/README.md`](../scripts/README.md) |
| [`tools/`](../tools) | repository integrity checks. See [`tools/README.md`](../tools/README.md) |
| [`tests/`](../tests) | the regression smoke test |
| [`data/datasets/`](../data/datasets) | the datasets that ship with the repository |
| [`legacy/`](../legacy) | earlier phases, kept reproducible. See [`legacy/README.md`](../legacy/README.md) |
| [`launchers/`](../launchers) | cluster submission scripts from the HPC phase |

## The current revision

The repository is mid-revision for *On the Minimization of Graph Counterfactual Explanations: Theory and a Local Bounded Search Algorithm*.

* [`revision/REVISION_EXPERIMENTS.md`](revision/REVISION_EXPERIMENTS.md) is the experiment plan: what each experiment answers, which reviewer comment it maps to, and what was deliberately not done.
* [`revision/REVISION_EXECUTION_ORDER.md`](revision/REVISION_EXECUTION_ORDER.md) is the run queue, maintained by `scripts/run_revision_queue.py`.
* [`revision/queue-runner.md`](revision/queue-runner.md) is the queue runner's cheat sheet.
* [`paper/`](paper) holds LaTeX fragments generated for the manuscript.

## Historical

* [`legacy/execution_pipeline.txt`](legacy/execution_pipeline.txt), the command-line workflow that predates the config-driven runners.
* [`legacy/configurations-status-v2.md`](legacy/configurations-status-v2.md), which GRETEL v2 configurations were known to work.

## One thing worth knowing before you change anything

A component's module path is part of its identity. `Context.get_name` hashes the resolved configuration, dotted class path included, into the name of every cache entry, saved artefact and result directory. Renaming a module that any configuration names will orphan the caches it produced and split its result tree in two.

`tools/README.md` explains which spellings of a reference feed that hash (imports, config `class` keys, and dotted strings passed to `get_class()` at runtime) and gives you the checks to run before and after a move.
