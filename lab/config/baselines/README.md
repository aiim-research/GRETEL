# `lab/config/baselines`

The published baselines, wired into the **current** pipeline.

Each of these methods already has a configuration under `legacy/config-v2/`, which is the generation it was published with and which still runs. Those configs use the older evaluator and its metric list. The configs here run the same method through `MainPipeline`, so its numbers land in `lab/output/results/` next to everything else and are directly comparable with the current matrix.

| Config | Method | Dataset |
|---|---|---|
| `asd_prand.jsonc` | pRand, probabilistic random perturbation | ASD + ASD custom oracle |
| `bbbp_maccs.jsonc` | MACCS, STONED over SELFIES | BBBP + GCN |

Run one like any other config:

```bash
python main.py lab/config/baselines/asd_prand.jsonc 1
```

`bbbp_maccs.jsonc` needs two optional packages that are not in `requirements-lock.txt`:

```bash
pip install exmol selfies
```

Budget time for it. MACCS runs a STONED search over SELFIES per instance, so a
full BBBP fold (2039 graphs) takes hours, not minutes: measured here at roughly
5 to 15 seconds per instance, 137 instances in 40 minutes. The per-instance
counterfactual dumps appear under the scope directory as the run proceeds, but
`results_<fold>_<run>.json` is only written when the fold finishes. To see the
method work without waiting, use `python tests/catalogue_smoke.py --only maccs`,
which stops after one instance.

## Baselines not here, and why

**iRand, DCE, OFS, RSGG, OBS, DBS** are already first-class citizens of the current matrix under `lab/config/generate_minimize/`, so they need nothing extra.

**DDBS** is in the current matrix too, in its decoupled form: `dfs` as generator plus `dbs` as minimizer. The monolithic implementation, `src/explainer/heuristic/ddbs.py`, keeps its v2 config. Note that `src/explainer/future/heuristic/ddbs.py` is an empty file and always has been, so there is no current-generation wrapper for the monolithic version.

**MEG** and **CounteRGAN** have current-generation wrappers but cannot be run: both fail on a torch device mismatch, the model landing half on `cuda` and half on `cpu`. The failure predates this repository's reorganisation, reproduced against a checkout of `main`. `tests/catalogue_smoke.py` tracks them in `KNOWN_BROKEN`. Adding configs for them would only add configs that crash, so they wait for the device fix.

**EAGER**, **COMBINEX** and **MOExp** have no `future/` wrapper, so they cannot be named by a current-generation config at all. EAGER additionally shares the device bug. Their v2 configurations are `legacy/config-v2/EAGER/{asd,bbbp,tcr28}.json` and `legacy/config-v2/new_datasets/AIDS_COMBINEX.json`; MOExp has never had a configuration.

`python tests/catalogue_smoke.py` runs one instance through each baseline from its own v2 config, which is the cheapest way to see which of them still work.
