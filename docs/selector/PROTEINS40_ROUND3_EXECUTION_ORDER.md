# Selector queue, PROTEINS round 3: seeds 1 and 2 for the frozen v3b arms

Auto-managed by `scripts/run_revision_queue.py`. Same protocol as round 2 (`PROTEINS40_ROUND2_EXECUTION_ORDER.md`). With these, v3b, v3b-frozen, v3b-pre-frozen, v2e and v2e-random all have seeds 0, 1 and 2 on the same 40 instances, so the effect of online training (v3b against v3b-frozen) and of pretraining (v3b-pre-frozen against v3b-frozen and v2e) can be tested with `analyze_selector_runs.py seeds`.

```
nohup ~/miniconda3/envs/GRTL/bin/python scripts/run_revision_queue.py --list docs/selector/PROTEINS40_ROUND3_EXECUTION_ORDER.md --workers 1 --gpu --timeout 21600 > lab/output/queue_proteins40_r3.log 2>&1 &
```

- [ ] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v3b-pre-frozen-s1/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v3b-frozen-s1/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v3b-pre-frozen-s2/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v3b-frozen-s2/generate_minimize0.jsonc
