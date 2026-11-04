# Selector queue: PROTEINS, 10 folds, 10000 oracle calls (thesis protocol)

Auto-managed by `scripts/run_revision_queue.py`. Configs from `scripts/gen_selector_proteins_folds.py` (its docstring describes the arms). Every fold's full test split, minimizer seed 0, DCE generator shared through `lab/data/cache/generator_outputs`, `max_oracle_calls` 10000. Arms: plain LBS (`strict_budget`), the selector (formerly v3b-pre-frozen: pretrained on the v3-collect examples and frozen), the same network untrained, and the same skeleton with uniform proposals. Two workers (the frozen and random arms never write checkpoints, so they can run side by side).

Every improvement is now logged with the oracle calls spent so far (`Found solution with size: N, calls: k`), so the per-instance trajectories can be compared (instances where plain LBS stalls at a large size while the selector converges early).

```
nohup ~/miniconda3/envs/GRTL/bin/python scripts/run_revision_queue.py --list docs/selector/PROTEINS_FOLDS_EXECUTION_ORDER.md --workers 2 --gpu --timeout 172800 > lab/output/queue_proteins_folds.log 2>&1 &
```

- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-untrained/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-random/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls/generate_minimize1.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel/generate_minimize1.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-untrained/generate_minimize1.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-random/generate_minimize1.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls/generate_minimize2.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel/generate_minimize2.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-untrained/generate_minimize2.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-random/generate_minimize2.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls/generate_minimize3.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel/generate_minimize3.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-untrained/generate_minimize3.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-random/generate_minimize3.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls/generate_minimize4.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel/generate_minimize4.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-untrained/generate_minimize4.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-random/generate_minimize4.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls/generate_minimize5.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel/generate_minimize5.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-untrained/generate_minimize5.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-random/generate_minimize5.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls/generate_minimize6.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel/generate_minimize6.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-untrained/generate_minimize6.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-random/generate_minimize6.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls/generate_minimize7.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel/generate_minimize7.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-untrained/generate_minimize7.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-random/generate_minimize7.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls/generate_minimize8.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel/generate_minimize8.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-untrained/generate_minimize8.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-random/generate_minimize8.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls/generate_minimize9.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel/generate_minimize9.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-untrained/generate_minimize9.jsonc
- [ ] lab/config/generate_minimize/proteins-sel/dce/dce-lcls-sel-random/generate_minimize9.jsonc
