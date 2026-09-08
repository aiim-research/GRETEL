# Selector v2 execution queue (Synthie + BBBP, DCE generator)

Auto-managed by `scripts/run_revision_queue.py` (same format as `REVISION_EXECUTION_ORDER.md`). Each `- [ ]` is a config; the runner flips it to `- [x]` once its results JSON appears under `lab/output/results/<scope>/` and unchecks it if results vanish.

Purpose: evaluate the second version of the online edge selector (`local_search_selection_net_v2.LocalSearch`, `Tagging/OnlineSelectorV2.py`) against (a) plain LBS, whose results already exist under `lab/output/results/{synthie,bbbp}_dce_lcls/`, (b) the controlled ablation `*-v2-random` (identical search skeleton, uniform random proposals, no learning), and (c) the first selector version (`dce-lcls-net`, thesis code).

Protocol: DCE generator with its fixed internal seed, minimizer seed 0, fold_id per file (0..9). Run **strictly in series** (`--workers 1`): the learned selector persists its checkpoint per dataset in `models/edge_selector_v2_<dataset>.pt` and later folds must see what earlier folds learned. Do not delete that checkpoint between folds. Delete it (and the scope's results) if you want a cold start.

Start / resume from the repo root:
```
python scripts/run_revision_queue.py --list SELECTOR_V2_EXECUTION_ORDER.md --workers 1
# GPU:
python scripts/run_revision_queue.py --list SELECTOR_V2_EXECUTION_ORDER.md --workers 1 --gpu
# Sync checkboxes with disk and exit:
python scripts/run_revision_queue.py --list SELECTOR_V2_EXECUTION_ORDER.md --sync-only
```

Order: Synthie first (smaller), learned v2 before its random ablation, BBBP after. The v1 selector runs last because it reuses the already trained thesis checkpoints in `models/edge_selector_<dataset>.pt` (warm start), so it is not a cold-start comparison.

Total: 100 configs.


## Tier 1 - Synthie, selector v2 learned (10 configs)

- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2/generate_minimize1.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2/generate_minimize2.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2/generate_minimize3.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2/generate_minimize4.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2/generate_minimize5.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2/generate_minimize6.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2/generate_minimize7.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2/generate_minimize8.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2/generate_minimize9.jsonc

## Tier 2 - Synthie, same skeleton with random proposals (ablation) (10 configs)

- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2-random/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2-random/generate_minimize1.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2-random/generate_minimize2.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2-random/generate_minimize3.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2-random/generate_minimize4.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2-random/generate_minimize5.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2-random/generate_minimize6.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2-random/generate_minimize7.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2-random/generate_minimize8.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2-random/generate_minimize9.jsonc

## Tier 3 - BBBP, selector v2 learned (10 configs)

- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2/generate_minimize1.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2/generate_minimize2.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2/generate_minimize3.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2/generate_minimize4.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2/generate_minimize5.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2/generate_minimize6.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2/generate_minimize7.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2/generate_minimize8.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2/generate_minimize9.jsonc

## Tier 4 - BBBP, same skeleton with random proposals (ablation) (10 configs)

- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2-random/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2-random/generate_minimize1.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2-random/generate_minimize2.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2-random/generate_minimize3.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2-random/generate_minimize4.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2-random/generate_minimize5.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2-random/generate_minimize6.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2-random/generate_minimize7.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2-random/generate_minimize8.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2-random/generate_minimize9.jsonc

## Tier 5 - selector v1 (thesis code, warm-started from existing checkpoints) (20 configs)

- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net/generate_minimize1.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net/generate_minimize2.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net/generate_minimize3.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net/generate_minimize4.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net/generate_minimize5.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net/generate_minimize6.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net/generate_minimize7.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net/generate_minimize8.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net/generate_minimize9.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net/generate_minimize1.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net/generate_minimize2.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net/generate_minimize3.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net/generate_minimize4.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net/generate_minimize5.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net/generate_minimize6.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net/generate_minimize7.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net/generate_minimize8.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net/generate_minimize9.jsonc

## Tier 6 - selector v2b (ranked block removal, no veto, no swap add-positives, more add updates) (20 configs)

- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2b/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2b/generate_minimize1.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2b/generate_minimize2.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2b/generate_minimize3.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2b/generate_minimize4.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2b/generate_minimize5.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2b/generate_minimize6.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2b/generate_minimize7.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2b/generate_minimize8.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2b/generate_minimize9.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2b/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2b/generate_minimize1.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2b/generate_minimize2.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2b/generate_minimize3.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2b/generate_minimize4.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2b/generate_minimize5.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2b/generate_minimize6.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2b/generate_minimize7.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2b/generate_minimize8.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2b/generate_minimize9.jsonc

## Tier 7 - v2b skeleton with random proposals (ablation for v2b) (20 configs)

- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2b-random/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2b-random/generate_minimize1.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2b-random/generate_minimize2.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2b-random/generate_minimize3.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2b-random/generate_minimize4.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2b-random/generate_minimize5.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2b-random/generate_minimize6.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2b-random/generate_minimize7.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2b-random/generate_minimize8.jsonc
- [ ] lab/config/generate_minimize/synthie/dce/dce-lcls-net-v2b-random/generate_minimize9.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2b-random/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2b-random/generate_minimize1.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2b-random/generate_minimize2.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2b-random/generate_minimize3.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2b-random/generate_minimize4.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2b-random/generate_minimize5.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2b-random/generate_minimize6.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2b-random/generate_minimize7.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2b-random/generate_minimize8.jsonc
- [ ] lab/config/generate_minimize/bbbp/dce/dce-lcls-net-v2b-random/generate_minimize9.jsonc
