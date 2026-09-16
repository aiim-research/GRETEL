# Repository maintenance tools

Static checks that guard the one invariant this repository cannot afford to lose: **a module path is part of an experiment's identity**.

`Context.get_name` builds every cache, artifact and result directory name by MD5-hashing the object's `local_config`, and `local_config` contains the dotted class path verbatim. So `ASD-15273954d84e872cf0b021cd4477bfdc` is the hash of a payload that literally starts with `ASD_class=src.dataset.dataset_base.Dataset_parameters_generator_class=src.dataset.generators.asd.ASD...`. Rename `src/dataset/generators/asd.py` and that directory name changes: the cached dataset, the trained oracle, the DCM medoids and the result tree all stop being found, and a re-run silently writes a second copy next to the first.

Three kinds of reference feed that hash, and all three have to be kept in sync:

| Where | Looks like |
|---|---|
| Python imports | `from src.explainer.future.search.dces import DCESExplainer` |
| Config `class` keys | `"class": "src.explainer.future.search.dcm.DCM"` |
| Python string literals | `set_proto_kls('src.explainer.legacy.generative.gans.graph.model.GAN')` |

The third is the one that bites: those strings are resolved by `get_class()` at runtime and injected into `local_config` as defaults, so they are hashed exactly like the config ones, but no import analysis sees them.

What is **not** hashed: `compose_*` snippet paths and `store_paths` addresses. The composer resolves them into the config before anything is hashed, so moving `lab/config/snippets/...` or the cache root is safe.

## The tools

### `check_config_refs.py`
Scans every `.json` / `.jsonc` in the repository, resolves each `src.*` class reference against the actual files, and checks that each referenced config path exists. Fails if the number of broken references grows over `config_refs_baseline.json`.

```
python tools/check_config_refs.py
python tools/check_config_refs.py . /tmp/report.json   # full per-config report
```

### `module_reachability.py`
Classifies every module under `src/` as ACTIVE, LEGACY-ONLY or ORPHAN by walking all three reference kinds from the current experiment surface. Use it before moving anything: **anything ACTIVE keeps its module path.**

```
python tools/module_reachability.py [report.json]
```

### `import_smoke.py`
Imports every module under `src/` and reports failures, ignoring optional heavy dependencies and the pre-existing breakage listed in `import_smoke_baseline.txt`.

```
python tools/import_smoke.py
python tools/import_smoke.py src/explainer
```

### `move_module.py`
Moves modules and rewrites every reference to them (imports, config `class` keys, string literals, and the `src/a/b.py` spelling used in docs), then `git mv`s the files and creates any missing `__init__.py`.

```
python tools/move_module.py plan.tsv            # dry run
python tools/move_module.py plan.tsv --apply
```

`plan.tsv` is one `old.dotted.path<TAB>new.dotted.path` per line.

## Before and after any move

```
python tools/check_config_refs.py && python tools/import_smoke.py
```

Both must stay green.
