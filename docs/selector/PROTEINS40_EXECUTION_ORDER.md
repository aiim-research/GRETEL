# Selector queue: PROTEINS, 40 random instances, 2000 minimizer oracle calls

Auto-managed by `scripts/run_revision_queue.py`. Each `- [ ]` is a config, flipped to `- [x]` once `lab/output/results/<scope>/.../results_0_1.json` exists.

Protocol: fold 0, the 40 ids of `lab/config/snippets/proteins_40_instances.json`, DCE generator with its fixed internal seed (same starting counterfactual for every method), minimizer seed 0, `max_oracle_calls` 2000 for every method (plain LBS with `strict_budget`). Every learned selector starts cold: the checkpoints from the previous machine are parked in `models/prev-machine-2026-09-08/`. One worker, in series (the machine is shared), with `--gpu` because the PROTEINS oracle cache was written from a CUDA session and does not load with CUDA hidden. Analyse with `scripts/analyze_selector_runs.py stats`.

```
nohup ~/miniconda3/envs/GRTL/bin/python scripts/run_revision_queue.py --list docs/selector/PROTEINS40_EXECUTION_ORDER.md --workers 1 --gpu --timeout 21600 > lab/output/queue_proteins40.log 2>&1 &
```

Order: the references first (plain LBS, thesis selector v1), then the v2 variants paired with their random ablation, most promising first.

- [ ] lab/config/generate_minimize/proteins40/dce/dce-lcls/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/proteins40/dce/dce-lcls-net/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v2g/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v2g-random/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v2e/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v2e-random/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v2f/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v2f-random/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v2b/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v2b-random/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v2/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v2c/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v2c-random/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v2d/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v2d-random/generate_minimize0.jsonc
