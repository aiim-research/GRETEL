# Selector queue, v4 (PROTEINS)

Auto-managed by `scripts/run_revision_queue.py`. Step 1: `v4-collect` repeats the uniform-proposal collection on the 72 fold-0 test instances outside the evaluation set (same skeleton as v2e), now storing with every example its within-instance evidence block (`OnlineSelectorV2.evidence_block`, computed from the moves evaluated before it) and the oracle-call step, for the offline test of the v4 features (`scripts/selector_pretrain_bench.py --evidence`). Offline (instance-split, 4 folds, and the 18-instance validation of the pretraining script) the evidence block lifts the held-out removal AUC of the pretrained network from 0.756 to 0.828 and its skill from +0.16 to +0.26. Step 2: the v4 arms (configs from `scripts/gen_selector_v4_configs.py`, checkpoint `models/pretrained/edge_selector_v4-pre_PROTEINS.pt`), seeds 0, 1, 2 each: v4 (evidence + adaptive sweep temperature), v4-ev (evidence only), v4-adapt (adaptive temperature on the v3b checkpoint). Reference: v3b-pre-frozen with the same seeds.

```
nohup ~/miniconda3/envs/GRTL/bin/python scripts/run_revision_queue.py --list docs/selector/PROTEINS40_V4_EXECUTION_ORDER.md --workers 1 --gpu --timeout 21600 > lab/output/queue_proteins40_v4.log 2>&1 &
```

- [x] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v4-collect/generate_minimize0.jsonc
- [x] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v4/generate_minimize0.jsonc
- [x] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v4-ev/generate_minimize0.jsonc

Round 2 (2026-10-02): v4 and v4-ev came out identical on seed 0 (the share-of-tried temperature never moved) and both lost GED against v3b-pre-frozen (3/22/15, Holm 0.036) at the same calls. The remaining v4 / v4-ev / v4-adapt seeds were dropped. Now: v4-adapt2 (temperature doubling per failed sweep on a state), an on-policy collection with the v3b-pre-frozen policy, and v4-ev2 (evidence pretrained on uniform plus on-policy examples, checkpoint written after the collection).

- [x] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v4-adapt2/generate_minimize0.jsonc
- [ ] lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v4-onpolicy-collect/generate_minimize0.jsonc
