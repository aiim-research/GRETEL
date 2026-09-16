# `src/legacy`

Careful: this directory is **not** entirely retired.

## Still live

`explainer/rsgg_v2/` is the RSGG implementation the current pipeline runs.
`src/explainer/future/generative/rsgg.py` subclasses it:

```python
from src.legacy.explainer.rsgg_v2.generative.rsgg import RSGG as RSGGOld

class RSGG(RSGGOld, metaclass=ExplainerTransformMeta):
    pass
```

Its module path is therefore hashed into the name of every RSGG cache entry
and result directory the revision batch produces, and into the GAN defaults
that `src/legacy/explainer/rsgg_v2/generative/gans/graph/model.py` injects into `local_config`. The
name is a historical accident (it was moved here when a newer RSGG was
expected) but it cannot be changed without invalidating those artefacts. See
`tools/README.md`.

Only `rsgg_v2/generative/gans/image/` inside it is dead.

## Retired

* `explainer/generative/` - the RSGG generation before `rsgg_v2`, with its own
  GAN stack, plus `per_cls_explainer.py`.
* `evaluation/future/` - the metric classes from the evaluator design that the
  stage pipeline (`src/evaluation/future/stages/`) replaced. Reached only from
  `lab/config/snippets/default_metrics.json`, which the retired configs
  compose. Their imports still said `src.evaluation.future.metrics` from
  before the move; fixed so the retired configs resolve again.
* `data_analysis/` - notebook-era result aggregation, superseded by
  `scripts/_results_agg.py` and `lab/notebooks/stats_visualizer.ipynb`.

## Why not reorganise further

Moving anything under `explainer/rsgg_v2/` changes hashes. The rest is already
where a reader expects retired code to be, so it stays put and this file says
which is which.
