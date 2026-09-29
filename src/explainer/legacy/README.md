# `src/explainer/legacy`

What is left here is retired: nothing imports it, no configuration names it,
and no published method depends on it.

| Module | Why it is here |
|---|---|
| `explainer_stub.py` | a no-op explainer used while wiring the factory, formerly `src/stubs/` |
| `helpers/gcn.py`, `helpers/gcn_perturb.py` | the CF-GNNExplainer perturbation layers. Nothing imports them: the CF2 implementation in `src/explainer/generative/cf2.py` carries its own |
| `helpers/caching.py` | an explainer cache that imports `clean_cfg` from a module it never lived in |

## What is NOT here, and why

The published methods that only the retired configurations under
`legacy/config-v2/` reach are **not** legacy. MEG, MACCS, pRand, DDBS,
CounteRGAN, EAGER, COMBINEX, the first RSGG and its GAN stack are the
comparison baselines the framework exists to offer, and the README advertises
them. They live where they always did:

    src/explainer/generative/   cf2, clear, rsgg, eager, gcountergan, gans/
    src/explainer/search/       dces, i_rand, maccs, p_rand, moexp
    src/explainer/heuristic/    obs, obs_dist, ddbs
    src/explainer/rl/           meg, meg_utils/
    src/explainer/learned/      combinex, graph_perturber, perturber/
    src/explainer/per_cls_explainer.py

That directory is the implementation layer of the framework. Several of its
modules are subclassed by `src/explainer/future/`, which adds the
`Explanation` wrapper, and the rest are reachable by naming them in a config.
Being old is not the same as being retired.
