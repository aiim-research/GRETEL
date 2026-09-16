# `obs-plus` tagging configs (unrunnable)

2660 configs, the `*-obs-plus` variant of every (tagger, dataset, generator)
combination in `lab/config/tagging/`. All of them declare

```json
"minimizer": { "class": "src.explainer.future.meta.minimizer.random-plus.RandomPlus" }
```

`RandomPlus` does not exist and never did: no such class appears anywhere in
the repository's history, and `random-plus` could not be imported even if the
module existed, because a hyphen is not legal in a Python module name. Every
one of these configs raises on load.

The matching `*-random-plus` directories under
`lab/config/generate_minimize/` were removed in `c85c4362e` ("Clean up
generate_minimize tree and standardize variant set"); the tagging tree was
missed, so these stayed behind and made up 13% of it.

They are kept rather than deleted in case the variant is revived. To bring one
back, restore the directory and point `minimizer.class` at a minimizer that
exists (`src/explainer/future/meta/minimizer/` has `obs.py`, `dbs.py` and
`dummy.py`) or implement the intended one.

The structure under here mirrors the original:
`<tagger>/<dataset>/<generator>/<generator>-obs-plus/`.
