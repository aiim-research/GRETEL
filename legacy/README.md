# `legacy/`

Material from earlier phases of GRETEL, kept so the published work stays
reproducible. Nothing here is part of the current revision batch.

## `legacy/config-v2/`

The configuration tree GRETEL v2 shipped with: the experiments behind the
CIKM'22 and WSDM'23 papers and the ACM Computing Surveys survey, plus the JMLR
comparison, the EAGER and RSGG-CE reference configs and the ensemble studies.
It was the repository's top-level `config/` directory.

These configs name the explainer classes that now live under
`src/explainer/legacy/` and the metrics under `src/legacy/evaluation/`; both
sets of references were updated when those modules moved, so the configs still
resolve. Run one the same way as any other:

```
python main.py legacy/config-v2/<config>.jsonc
```

Caveat: a handful still reference classes that were deleted rather than moved
(`src.explainer.ensemble.*`, `src.evaluation.stages.*`,
`src.embedder.newgraph2vec`). `python tools/check_config_refs.py` lists them.

Not to be confused with `lab/config/`, which is where the current experiments
live.

## `legacy/examples/`

The GRETEL v1 tutorial notebooks: evaluating explainers, adding datasets,
adding explainers, adding an evaluation metric, visualising explanations. They
target an API that has since changed, so treat them as documentation of the
original design rather than as runnable examples. The 83 MB `tutorial.mp4` that
accompanied them is no longer versioned; `git log --all -- legacy/examples/tutorial.mp4`
finds it in the history.
