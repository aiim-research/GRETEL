# `src/` - the framework

Read this before adding code. It covers how a component is wired, where each kind of component lives, and the one rule that is easy to break by accident.

## The rule

**A module's path is part of its identity.** `Context.get_name` hashes a component's resolved configuration, dotted class path included, into the name of every cache entry, saved artefact and result directory it produces. `ASD-15273954d84e872cf0b021cd4477bfdc` is the MD5 of a payload that starts with `ASD_class=src.dataset.dataset_base.Dataset_parameters_generator_class=src.dataset.generators.asd.ASD...`.

So moving or renaming a module that any configuration names will orphan every cache it produced and split its result tree in two. New code can go wherever it belongs. Existing code cannot be tidied casually. [`tools/README.md`](../tools/README.md) has the full story and the checks to run.

## How a component is wired

There is no registry and nothing to import in a central file. A component is a class named by its dotted path in a config, plus a parameter dictionary:

```jsonc
"explainer": {
  "class": "src.explainer.future.search.dces.DCESExplainer",
  "parameters": { "epochs": 500 }
}
```

A factory calls `get_class(snippet["class"])(context, snippet)` and that is the whole mechanism. Write the class, name it in a config, done.

### Lifecycle

Every component inherits from `Configurable`, whose constructor runs two hooks in order:

1. **`check_configuration()`** fills in defaults. This is where you write them, because whatever lands in `local_config` here is what gets hashed. Always call `super().check_configuration()` first.
2. **`init()`** builds the object from the now-complete `local_config`.

Components that persist something extend `Savable` (adds `load_or_create()`, which reads the artefact if it exists and calls `create()` and `write()` otherwise) or `Trainable` (a `Savable` whose `create()` is `fit()`, wrapping your `real_fit()` with timing and a `retrain` flag).

```
Base                          name, context, __str__
 └── Configurable             check_configuration() + init()
      ├── Explainer           explain(instance)
      ├── Generator           generate_dataset()
      ├── BaseManipulator     node_info() / edge_info() / graph_info()
      ├── Stage               process(explanation)
      │    └── MetricStage    plus aggregate() over instances
      ├── ExplanationAggregator
      └── Savable             load_or_create(), read(), write(), create()
           └── Trainable      real_fit(), retrain
                ├── Oracle    predict(), counts oracle calls for you
                ├── Embedder
                ├── TorchBase torch models
                └── ExplanationMinimizer   minimize(explanation)
```

An explainer that trains inherits from both sides, as `ExplainerEnsemble(Explainer, Trainable)` does.

Two conveniences worth knowing: `Oracle.predict` is `@final` and increments the call counter, so the Oracle Calls metric works without any effort from you, and `init_dflts_to_of(self.local_config, key, "src.some.Class")` fills in a whole nested sub-component with its own defaults.

## Package map

| Package | What it holds |
|---|---|
| `core/` | the extension contracts: `Configurable`, `Savable`, `Trainable`, `Explainer`, `Oracle`, `Embedder`, the factory helpers. Start here to understand the machinery |
| `dataset/` | `dataset_base.py`, `generators/` (one per data source), `manipulators/` (feature computation applied after loading), `instances/` (`GraphInstance`) |
| `oracle/` | `nn/` (GCN and the torch wrapper), `tabulars/` (KNN, SVM), `custom/` (exact oracles for synthetic data) |
| `embedder/` | graph2vec, molecular fingerprints |
| `explainer/` | the explanation methods. See the note below |
| `evaluation/` | `evaluation/future/evaluator.py` plus `evaluation/future/stages/`, one file per metric |
| `future/explanation/` | the `Explanation` types that flow through the pipeline |
| `utils/` | context, config helpers, composer, seeding, metrics, torch helpers |
| `LLMexplaneability/` | natural-language explanation of counterfactuals (the directory name is a typo, kept because configs name it) |

Anything under a `legacy/` subdirectory is retired: reachable only from retired configs, or from nothing. Each has a README saying what it was. Do not build on it.

## The `future/` trap

`src/explainer/future/` is the **current** code, not future work. It is the generation that produces `Explanation` objects. The modules outside it that are not under `legacy/` are the older implementations that the `future/` ones still subclass:

```python
# src/explainer/future/search/dces.py
from src.explainer.search.dces import DCESExplainer as DCESExplainerOld

class DCESExplainer(DCESExplainerOld, metaclass=ExplainerTransformMeta):
    pass
```

The metaclass wraps `explain()` so a bare counterfactual instance comes back as a `LocalGraphCounterfactualExplanation`. That is the only difference. `src/explainer/legacy/README.md` lists exactly which outside-`future/` modules are still live for this reason.

The name is a historical accident and cannot be fixed: renaming it would change every hash. Write new explainers under `future/`.

## Where to put your contribution

| You are adding | Put it in | Subclass | Named in the config as |
|---|---|---|---|
| a dataset | `dataset/generators/` | `Generator` | `dataset.parameters.generator.class` |
| a feature computed per graph | `dataset/manipulators/` | `BaseManipulator` | `dataset.parameters.manipulators[].class` |
| a classifier to explain | `oracle/nn/` or `oracle/tabulars/` | `Oracle` | `oracle.class` |
| an explanation method | `explainer/future/<family>/` | `Explainer` | `explainer.class` |
| a counterfactual minimizer | `explainer/future/meta/minimizer/` or `.../metaheuristic/` | `ExplanationMinimizer` | `explainer.parameters.minimizer.class` |
| an evaluation metric | `evaluation/future/stages/` | `MetricStage` | the `stages` list of the pipeline snippet |
| an ensemble aggregator | `explainer/future/ensemble/aggregators/` | `ExplanationAggregator` | `explainer.parameters.aggregator.class` |

The explainer families under `future/` are `search/` (searches the dataset or the graph directly), `heuristic/`, `metaheuristic/` (local search and its variants), `generative/`, `rl/`, `ensemble/`, `meta/` (composes other explainers, for example `GenerateMinimize`).

## A worked example: a new explainer

```python
# src/explainer/future/search/my_method.py
from src.core.explainer_base import Explainer
from src.dataset.instances.graph import GraphInstance
from src.future.explanation.local.graph_counterfactual import (
    LocalGraphCounterfactualExplanation,
)


class MyMethod(Explainer):

    def check_configuration(self):
        super().check_configuration()
        # Defaults belong here: local_config is what gets hashed, so two runs
        # that rely on the same default share a cache entry.
        p = self.local_config["parameters"]
        p["max_oracle_calls"] = p.get("max_oracle_calls", 2000)

    def init(self):
        super().init()
        self.max_oracle_calls = self.local_config["parameters"]["max_oracle_calls"]

    def explain(self, instance: GraphInstance):
        # self.dataset and self.oracle are injected by the factory.
        # Every self.oracle.predict() call is counted for you.
        counterfactual = ...
        return LocalGraphCounterfactualExplanation(
            context=self.context,
            dataset=self.dataset,
            oracle=self.oracle,
            explainer=self,
            input_instance=instance,
            counterfactual_instances=[counterfactual],
        )
```

Then copy an existing config, for example `lab/config/base/bbbp_gcn_irand.jsonc`, point `explainer.class` at `src.explainer.future.search.my_method.MyMethod`, and run it:

```bash
python main.py lab/config/my_method.jsonc 1
```

If your method needs to train or to persist anything, subclass `Trainable` instead and implement `real_fit()`, `read()` and `write()`. `load_or_create()` then handles caching, and the artefact is keyed by your configuration automatically.

## Before you open a pull request

```bash
python tools/import_smoke.py            # nothing became unimportable
python tools/check_config_refs.py       # no configuration lost a reference
python tests/regression_smoke.py --datasets asd --timeout 240
```

Both tools carry a baseline of pre-existing damage and fail only on new breakage. If you moved a module rather than adding one, use `tools/move_module.py`: it rewrites all three spellings of a reference, including the dotted strings in Python that plain import analysis never sees.
