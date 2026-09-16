# `src/explainer/legacy`

The explainer generation that predates `src/explainer/future/`. Every module
here is reachable only from the retired configs under `legacy/config-v2/` (the
CIKM/WSDM/survey-era experiments) or from nothing at all. The current revision
batch does not touch any of it.

| Subtree | What it is |
|---|---|
| `generative/rsgg.py`, `generative/gans/` | the first RSGG and its GAN stack. The RSGG the live pipeline runs is `src/legacy/explainer/rsgg_v2/`, wrapped by `src/explainer/future/generative/rsgg.py` |
| `generative/eager.py` | EAGER, on the learnable-edges GAN |
| `generative/gcountergan.py` | the CounteRGAN port for graphs |
| `per_cls_explainer.py` | per-class base the three generative explainers above share |
| `heuristic/ddbs.py` | data-driven bidirectional search, before the generator/minimizer split |
| `search/maccs.py`, `search/p_rand.py`, `search/moexp.py` | molecule-oriented and random baselines |
| `learned/`, `helpers/` | COMBINEX and the CF-GNNExplainer perturbation helpers |
| `rl/` | MEG and its molecule environment |
| `explainer_stub.py` | a no-op explainer, formerly `src/stubs/` |

## What stayed behind in `src/explainer/`

These are **not** legacy despite living outside `future/`: the `future/`
modules are thin subclasses of them, so they run in every current experiment
and their module paths are hashed into today's cache and result names.

    explainer_factory.py
    generative/cf2.py          <- future/generative/cf2.py
    generative/clear.py        <- future/generative/clear.py
    heuristic/obs.py           <- future/heuristic/obs.py
    heuristic/obs_dist.py      <- named by future/search/{ofs,dfs}.py
    search/dces.py             <- future/search/dces.py
    search/i_rand.py           <- future/search/i_rand.py
    rl/meg_utils/utils/molecular_instance.py  <- src/dataset/generators/mol_gen.py

`molecular_instance.py` is the last survivor of the MEG utilities: it defines
`MolecularInstance`, which the BBBP/HIV dataset generators build. It kept its
path so instances already written to disk stay loadable.
