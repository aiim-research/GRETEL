# Selector queue, v4 (PROTEINS)

Auto-managed by `scripts/run_revision_queue.py`. Step 1: `v4-collect` repeats the uniform-proposal collection on the 72 fold-0 test instances outside the evaluation set (same skeleton as v2e), now storing with every example its within-instance evidence block (`OnlineSelectorV2.evidence_block`, computed from the moves evaluated before it) and the oracle-call step, for the offline test of the v4 features (`scripts/selector_pretrain_bench.py --evidence`). The v4 evaluation arms are appended once that test is done.

```
nohup ~/miniconda3/envs/GRTL/bin/python scripts/run_revision_queue.py --list docs/selector/PROTEINS40_V4_EXECUTION_ORDER.md --workers 1 --gpu --timeout 21600 > lab/output/queue_proteins40_v4.log 2>&1 &
```

- [ ] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v4-collect/generate_minimize0.jsonc
