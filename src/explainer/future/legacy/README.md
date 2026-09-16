# `src/explainer/future/legacy`

Modules from the `future/` explainer tree that nothing reaches any more: no
import, no config `"class"` key, no dotted string handed to `get_class()`.
They are kept rather than deleted because several are the only record of an
approach that was tried and dropped.

| Module | Why it is here |
|---|---|
| `generative/gcountergan.py` | GAN-based counterfactual port; superseded by RSGG |
| `heuristic/ddbs.py` | replaced by the `meta/minimizer` + generator split (`DFS` + `DBS`) |
| `metaheuristic/Tagging/ann.py` | approximate-nearest-neighbour tagger, needs `hnswlib` |
| `metaheuristic/Tagging/solution_explorer.py` | exploratory helper for the tagging experiments |
| `metaheuristic/local_search/local_search_ann.py` | the LocalSearch variant that used the ANN index above |
| `metaheuristic/manipulation/base.py` | abstract `Manipulator`; `methods.py` never subclassed it |
| `rl/meg.py` | MEG wrapper; the live MEG configs point at the pre-`future` class |
| `search/dces_multi.py`, `search/maccs.py`, `search/p_rand.py` | wrappers whose configs were retired |

The live tree next door (`search/`, `heuristic/`, `metaheuristic/`, `meta/`,
`generative/`, `ensemble/`, `cascade/`) is what the current experiments run.
