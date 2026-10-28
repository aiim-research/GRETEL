# Selector development study (archive)

Results and logs of the runs that led to the BLS selector, from the first calibration (2026-09-03) to the configuration used in the final 10-fold PROTEINS experiment (2026-10-06). The final experiment itself writes to `lab/output/results/proteins-sel_*` and is not here. The narrative, numbers and conclusions of every phase are in `docs/selector/SELECTOR_EXPERIMENTS_STATUS.md`.

Each phase has the same layout:

- `results/<scope>/.../results_<fold>_1.json`: per-instance metrics of each run, with the full resolved config inside (key `config`), so a run can be repeated even when its config file was temporary. From phase 2 on, also `cf_per_instance/fold_<f>/cf_<id>.json`: the instance and the counterfactual found for it.
- `queue_logs/<dataset>/<generator>-<minimizer>/fold<f>.log.gz`: the run log of each config, one folder per dataset, generator and minimizer as everywhere else in the repo (every improvement is a `Found solution with size: N, calls: k` line), and the runner logs `queue_logs/queue_*.log.gz`.
- `selector_logs/selector_<tag>_<dataset>.csv`: the selector's prequential monitor, one row per instance.

The analysis scripts find all of this without paths: a bare scope or monitor tag is looked up in `lab/output` first and then in every phase here, and a run log can be given as `<scope>:<fold>` (`scripts/result_paths.py`). For example `python scripts/analyze_selector_runs.py stats base=proteins40_dce_lcls sel=proteins40_dce_lcls-net-v3b-pre-frozen --ref base` or `python scripts/selector_trajectories.py proteins-sel_dce_lcls:0 proteins-sel_dce_lcls-sel:0`.

All runs use the caches committed in `lab/data/cache/` (same folds and oracles) and the DCE generator.

## 1-calibration (2026-09-03 to 2026-09-08)

Synthie folds 0 and 1, PROTEINS-10 (fold 0, ten instances) and BBBP: plain BLS against the thesis selector (v1) and the online variants v2 to v2g, each with a `-random` control. Status note sections 1 to 7. Configs: `lab/config/generate_minimize/{synthie,proteins10,bbbp}/dce/dce-lcls-net*`; queue: `docs/selector/SELECTOR_V2_EXECUTION_ORDER.md`.

- `results/` holds only the `results_*.json` files: the counterfactual dumps were not kept at the time.
- `results/_backup_precalibration/{synthie,bbbp}_dce_lcls` is plain BLS before its calibration.
- The selector checkpoints of this phase are `models/edge_selector_v2{b,e,f}_Synthie-*.pt`, `models/prev-machine-2026-09-08/` and `models/*.thesis-bak` (v1 as trained for the thesis).

## 2-proteins40 (2026-09-29 to 2026-10-02)

PROTEINS fold 0, forty instances, 2000 oracle calls per instance: plain BLS, v1, v2e, v2g, v3, v3b and v4 variants, each with frozen, random, pretrained (`-pre`) and seed (`-s1`, `-s2`) arms where they apply. Status note sections 8 and 8.1. Configs: `lab/config/generate_minimize/proteins40/`; queues: `docs/selector/PROTEINS40_EXECUTION_ORDER.md`, `PROTEINS40_ROUND2_EXECUTION_ORDER.md`, `PROTEINS40_ROUND3_EXECUTION_ORDER.md`, `PROTEINS40_V4_EXECUTION_ORDER.md`.

- `results/proteins72collect_*`: the collection runs on 72 other instances of the fold, whose moves trained the selector offline.
- `pretrain/`: the reports of `scripts/pretrain_selector_v3.py` and of the ranker benches (`scripts/selector_pretrain_bench.py`, `scripts/selector_offline_bench.py`). The collections themselves (`lab/output/selector_pretrain/*_collect.pt`, 10 to 15 MB) are not tracked. The pretrained selector they produced is `models/pretrained/edge_selector_v3-pre_PROTEINS.pt`.
- `fold_instances/`: the ids and sizes of the 112 test instances of fold 0, from which the forty (and the 72 of the collection) were drawn.
- The online selectors' checkpoints (`models/edge_selector_v*_PROTEINS-*.pt`) are not tracked.

## 3-multi40 (2026-10-01)

AIDS and COLORS-3, forty instances of fold 0 each: plain BLS, v1, v2e and v3b. The selector had no room to help on either dataset and both were dropped. Configs: `lab/config/generate_minimize/{aids40,colors340}/`; queue: `docs/selector/MULTI40_EXECUTION_ORDER.md`; `fold_instances/` has the instance ids and the logs of their selection.

## 4-final-config (2026-10-03 to 2026-10-06)

Tuning of the final selector on PROTEINS at 10000 calls (overshoot, no retries after a failed iteration, `stuck_swap_add`, `near_miss`). Status note section 8 and `docs/selector/SELECTOR_FAILURE_CASES.md`.

- `selector_logs/selector_smoke-*`: monitors of the smoke runs on a handful of instances (their results were not kept).
- `results/prelim_{ref,p1,p2,p12}`: the ten-instance preliminary test that chose the final configuration (p12), compared with `scripts/compare_prelim_logs.py`. Their configs were temporary and their logs were lost: the configs are inside the results files, the comparison is in the status note.
- `results/proteins-sel_dce_lcls`: plain BLS on fold 0 (complete) and part of fold 1, run locally before the experiment moved to the server, where it runs again. The final experiment's numbers come from the server run in `lab/output/results/`.
- `queue_logs/proteins-sel/dce-lcls-sel/fold0.log.gz`: the selector's fold 0 run with the configuration before p12, interrupted. Source of the failure cases. `queue_logs/proteins-sel/dce-lcls/` has the logs of the local plain BLS runs above.
- Queues: `docs/selector/PROTEINS_FOLDS_EXECUTION_ORDER.md`, `PROTEINS_FOLDS_SEL_EXECUTION_ORDER.md`, `PROTEINS_FOLDS_BASE_EXECUTION_ORDER.md`.
