# `src/explainer/future/legacy`

Modules from the `future/` explainer tree that nothing reaches: no import, no
config `"class"` key, no dotted string handed to `get_class()`. They are kept
rather than deleted because each is the only record of an approach that was
tried and dropped.

| Module | Why it is here |
|---|---|
| `metaheuristic/Tagging/ann.py` | approximate-nearest-neighbour tagger, needs `hnswlib` |
| `metaheuristic/Tagging/solution_explorer.py` | exploratory helper for the tagging experiments |
| `metaheuristic/local_search/local_search_ann.py` | the LocalSearch variant built on the ANN index above |
| `metaheuristic/manipulation/base.py` | abstract `Manipulator`; `methods.py` never subclassed it |
| `search/dces_multi.py` | a multi-counterfactual DCE variant that was never wired to a config |

The `future/` wrappers of the published baselines are **not** here: MEG,
MACCS, pRand, DDBS and CounteRGAN sit in `future/rl/`, `future/search/`,
`future/heuristic/` and `future/generative/` next to the methods the current
batch runs. See `src/explainer/legacy/README.md`.
