# Reproducing GRETEL experiments

Everything an experiment needs is declared in one configuration file. There is no hidden state: the same config run twice on the same code produces the same result directory, because that directory's name is a hash of the configuration itself.

This guide takes you from a fresh clone to a reproduced number.

## 1. Environment

The results were produced with Python 3.9, torch 2.5.1 on CUDA 12.1, and torch-geometric 2.6.1. Pick one route:

**With CUDA (the supported path).**

```bash
./scripts/setup-grtl-gpu.sh "$HF_TOKEN"
conda activate GRTL
```

The Hugging Face token is only needed for the LLM explainability module. Pass any non-empty string if you are not using it.

**With conda, CPU.**

```bash
conda env create -f environment.yml
conda activate GRETEL
```

**With pip, into an existing Python 3.9.**

```bash
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

`requirements-lock.txt` is a freeze of the environment that produced the published numbers and is the file to trust. `requirements.in` lists direct dependencies. `requirements.txt` is a pip-compile output resolved against a newer interpreter (it pins numpy 2.3.5 and torch 2.9.1) and will not install on Python 3.9.

Check the install:

```bash
python tools/import_smoke.py
python tools/check_config_refs.py
```

## 2. Data

Four datasets ship with the repository under `data/datasets/`: ASD and ADHD (brain networks), BBBP and HIV (molecules).

Tree-Cycles is generated procedurally from parameters in the config, so nothing is downloaded.

The TU datasets (PROTEINS, ENZYMES, BZR, AIDS, COLORS-3, Synthie, IMDB-BINARY, Cuneiform) are fetched by `torch_geometric` on first use into `data/working/<NAME>/`. That directory is not versioned. The first run of a config that needs one is slower.

## 3. What a configuration looks like

Take `lab/config/generate_minimize/asd/dce/dce-lcls/generate_minimize0.jsonc`: dataset ASD, generator DCE, minimizer LocalSearch, fold 0.

```jsonc
{
  "experiment": {
    "scope": "asd_dce_lcls",              // names the result directory
    "parameters": { "propagate": [ ... ] } // fold ids, manipulators, retrain flags
  },
  "doe-triplets": [{
    "compose_do": "./lab/config/snippets/do-pairs/ASD_ASD-Custom.json",  // dataset + oracle
    "explainer": {
      "class": "src.explainer.future.meta.generate_minimize.GenerateMinimize",
      "parameters": {
        "fold_id": 0,
        "generator": { "class": "src.explainer.future.search.dces.DCESExplainer", ... },
        "minimizer": { "class": "src.explainer.future.metaheuristic.local_search.local_search.LocalSearch", ... }
      }
    }
  }],
  "evaluator": { "class": "src.evaluation.future.evaluator.Evaluator",
                 "parameters": { "compose_pip": "./lab/config/snippets/minimizing_pipeline.json" } },
  "compose_strs": "./lab/config/snippets/default_store_paths.json"
}
```

Every key starting with `compose` is replaced by the contents of the file it names before anything else happens, so the shared pieces live once in `lab/config/snippets/` and each experiment file stays small. `propagate` pushes a parameter into several sections at once (the fold id into the explainer, `retrain: false` into the oracle, a manipulator set into the dataset).

`scope` is the only free-text label. Everything else about where output lands is derived.

## 4. Running one experiment

```bash
python main.py lab/config/generate_minimize/asd/dce/dce-lcls/generate_minimize0.jsonc 1
```

The second argument is the run number, written into the result filename.

`main.py` reads the config and picks the evaluation manager that matches it, so the same command runs a configuration from either generation. `future_main.py` runs the current generation too, with OMP and MKL thread caps applied before torch is imported, and is what the cluster launchers call.

Results land in

```
lab/output/results/<scope>/<dataset>/<oracle>/<explainer>/results_<fold>_<run>.json
```

where `<dataset>`, `<oracle>` and `<explainer>` are content hashes, for example `ASD-15273954d84e872cf0b021cd4477bfdc`. That hash is the MD5 of the component's resolved configuration, so two runs that agree on every parameter write to the same place and a run that differs anywhere writes somewhere else. It is what makes the store self-describing, and it is why module paths must not be renamed casually (`tools/README.md` has the full story).

Caches follow the same scheme under `lab/data/cache/`: build a dataset once and every later config that declares it identically reuses it.

## 5. Running a batch

**The revision batch.** `docs/revision/REVISION_EXECUTION_ORDER.md` is a checklist of 971 configs. The runner works through the unchecked ones, one subprocess each, and ticks a box when its results file appears:

```bash
python scripts/run_revision_queue.py --workers 8          # CPU
python scripts/run_revision_queue.py --workers 4 --gpu    # 2 workers per GPU
nohup python scripts/run_revision_queue.py --workers 4 --gpu > /tmp/queue.log 2>&1 &
```

It is restart-safe. On startup it checks the filesystem and skips anything already on disk, so interrupting it and running it again resumes where it stopped. Logs go to `lab/output/queue_logs/<scope>_fold<n>.log`.

That startup sync cuts both ways, and it is worth knowing before you run it: the checklist is rewritten to match what is on disk, in **both** directions. `lab/output/` is not versioned, so on a fresh clone, or on a second machine, or after the results are archived elsewhere, the sync finds nothing and unchecks all 971 entries in the tracked `REVISION_EXECUTION_ORDER.md`. That loses the record of what has already been run. Run the queue only where the results live, and if you only want to inspect the state, check `git diff` afterwards and revert the file if it was rewritten.

**An arbitrary matrix.** `scripts/run_experiments.py` runs a fixed set of combinations over all folds:

```bash
python scripts/run_experiments.py --datasets asd bzr --combos ofs/ofs-obs --folds 0 1 2
```

**A quick check that nothing is broken.** `tests/regression_smoke.py` runs one graph instance per combination and stops:

```bash
python tests/regression_smoke.py --datasets asd --timeout 240
```

## 6. Artefacts that must be trained first

Two generators load a pre-trained artefact instead of computing it per run. Train each once per dataset, before any config that uses it:

```bash
python scripts/compute_dcm.py --do-pair lab/config/snippets/do-pairs/ASD_ASD-Custom.json --proportion 1.0
python scripts/compute_lst_methods.py --do-pair lab/config/snippets/do-pairs/BZR_GCN.json --proportion 1.0
```

They write `DCM-*` and `LSTMethodsArtifact-*` into `lab/data/cache/explainers/<dataset-hash>/`. Both are small and several are versioned in the repository, so for the datasets used in the paper you can skip this step: the run will load the committed file. You will see `Loading: DCM-...` rather than `Creating: DCM-...` in the log.

## 7. The experiment matrix

`lab/config/generate_minimize/<dataset>/<generator>/<generator>-<minimizer>/generate_minimize<fold>.jsonc`

| Dimension | Values |
|---|---|
| dataset | `asd`, `bbbp`, `bbbp-no-attr`, `synthie`, `synthie-no-att`, `tcr-tco-300`, `tcr-gcn`, `proteins`, `enzymes`, `bzr`, `aids`, `colors-3`, `cuneiform`, `imdb`, and the two `tcr-ablation-*` sweeps |
| generator | `dce`, `ofs`, `dfs`, `rsgg`, `dcm` |
| minimizer | see below |
| fold | 0 to 9 |

| Minimizer variant | Class |
|---|---|
| `dummy` | `meta.minimizer.dummy.Dummy`, the generator's own counterfactual, unminimized. The baseline every other bar is compared against. |
| `lcls` | `metaheuristic.local_search.local_search.LocalSearch`, the paper's LBS |
| `obs` | `meta.minimizer.obs.OBS`, oblivious bidirectional search |
| `dbs` | `meta.minimizer.dbs.DBS`, data-driven bidirectional search |
| `rhc` | `metaheuristic.local_search.random_hill_climbing.RandomHillClimbing`, the budget-matched generic baseline |
| `lcls-var-1` .. `lcls-var-4` | strategy ablations: `var-1`/`var-2` drop a strategy, `var-3`/`var-4` permute the priority order |
| `lcls-net`, `lcls-trainable`, `lcls-ponderation`, `lcls-net-trainable` | learned edge-selector variants (these need the LST artefact) |
| `lcls-seed1` .. `lcls-seed3` | the same LBS configuration under different seeds, for the stability table |

The other config trees: `lab/config/tagging/` is the tagging-strategy study, `lab/config/ensembles/` and `lab/config/meta_ens/` the ensemble aggregation and explainer-selection work, `lab/config/bls_selection_net*/` the learned selector, `lab/config/base/` minimal single-explainer examples worth reading first.

## 8. Analysing results

`lab/notebooks/stats_visualizer.ipynb` builds the tables and figures from `lab/output/results/`. `scripts/_results_agg.py` is the same aggregation as an importable module, so a notebook and a script never disagree about what a cell means.

The aggregation is three steps, and the order matters: filter (GED and FED keep only correct counterfactuals, Oracle Calls and Correctness keep every record), then mean per fold, then mean over folds.

Figures:

```bash
python scripts/make_paper_figures.py                       # per-generator, into lab/graphics/
python scripts/make_paper_figures.py dce --out /path/to/paper/images
GRETEL_FIGURES_DIR=/path/to/paper/images python scripts/plot_final_results.py
```

Cells with no results yet are drawn as hatched placeholders, so the figures are usable while a batch is still running.

## 9. Which configurations belong to which paper

| Work | Configurations |
|---|---|
| Minimization of graph counterfactual explanations (under revision) | `lab/config/generate_minimize/`, queued by `docs/revision/REVISION_EXECUTION_ORDER.md`. The protocol and the reviewer-comment mapping are in `docs/revision/REVISION_EXPERIMENTS.md`. |
| Tagging strategies | `lab/config/tagging/` |
| Ensembles and explainer selection | `lab/config/ensembles/`, `lab/config/meta_ens/` |
| GRETEL v2 (CIKM'22, WSDM'23, the Computing Surveys survey, the JMLR comparison) | `legacy/config-v2/`, see `legacy/README.md` |

## 10. Reproducibility notes and known limits

**Seeding.** `src/utils/seeding.py` seeds `random`, numpy and torch from one value. The parameter name differs by component: `seed` for OFS, RSGG, LBS, OBS and RHC, `random_seed` for DFS and DBS. DCE takes none, being a deterministic search over the dataset.

Read `set_seed(None)` carefully: it is a **no-op**, deliberately, so that a config which never declared a seed keeps both its old behaviour and its old hash. Determinism therefore comes from the config stating a seed, not from a default. Every config in the matrix does state one: the scopes without a seed suffix set `0` explicitly on both generator and minimizer, and `dce-lcls-seed1` through `-seed3` set 1, 2 and 3. A config you write yourself that omits the parameter will not be reproducible.

**What is single-run and what is not.** The main matrix is run once per combination under a fixed seed. Only the LBS stability table repeats a configuration across seeds (`dce-lcls-seed1` through `-seed3`). This is deliberate: the generator's seed is fixed, so every minimizer sees the identical set of initial counterfactuals, and the only stochastic component left to measure is LBS itself.

**Correctness is a property of the generator, not of the minimizer.** A minimizer receives valid counterfactuals and shrinks them, so it cannot change how many were found. One known exception is tracked in `docs/revision/REVISION_EXPERIMENTS.md`: DBS sometimes returns a non-counterfactual, which lowers the figure.

**Budget accounting.** LBS checks `max_oracle_calls` at the top of its outer loop and can overshoot it within one pass, while RHC cuts exactly. When comparing the two at "the same budget", read the measured oracle calls rather than assuming the cap held.

**Retired configurations.** Some configs under `legacy/` reference classes that were deleted rather than moved and cannot run. `python tools/check_config_refs.py` lists every one of them and separates the live tree from the retired trees. The live tree is expected to stay at zero problems.
