# Selector queue, PROTEINS round 2: v3b arms and extra seeds

Auto-managed by `scripts/run_revision_queue.py`. Same 40 instances and protocol as `PROTEINS40_EXECUTION_ORDER.md` (fold 0, DCE, 2000 minimizer oracle calls, cold selectors), with the shared generator cache (`lab/data/cache/generator_outputs`; the DCE output is deterministic, so it is identical to the one the round-1 runs computed).

1. The v3b arms that separate the effect of training and of pretraining: v3b-frozen (same untrained network, same policy, never updated), v3b-pre (offline-pretrained on the 72 disjoint v3-collect instances, then online), v3b-pre-frozen.
2. Seeds 1 and 2 for v3b, v2e and v2e-random (configs from `scripts/gen_selector_seed_configs.py`), so that the comparisons rest on three seeds per method instead of one and seed-luck instances (ids 148, 159) stop dominating the means.

```
nohup ~/miniconda3/envs/GRTL/bin/python scripts/run_revision_queue.py --list docs/selector/PROTEINS40_ROUND2_EXECUTION_ORDER.md --workers 1 --gpu --timeout 21600 > lab/output/queue_proteins40_r2.log 2>&1 &
```

- [x] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v3b-frozen/generate_minimize0.jsonc
- [x] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v3b-pre/generate_minimize0.jsonc
- [x] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v3b-pre-frozen/generate_minimize0.jsonc
- [x] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v3b-s1/generate_minimize0.jsonc
- [x] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v2e-s1/generate_minimize0.jsonc
- [x] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v2e-random-s1/generate_minimize0.jsonc
- [x] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v3b-s2/generate_minimize0.jsonc
- [x] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v2e-s2/generate_minimize0.jsonc
- [x] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v2e-random-s2/generate_minimize0.jsonc
