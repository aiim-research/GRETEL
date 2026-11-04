# Selector queue: AIDS and COLORS-3, 40 random instances each, 2000 minimizer oracle calls

Auto-managed by `scripts/run_revision_queue.py`. Same protocol as `PROTEINS40_EXECUTION_ORDER.md`: fold 0, 40 instances drawn with `random.Random(2026)` from the fold-0 test split (`lab/config/snippets/aids_40_instances.json`, `colors3_40_instances.json`), DCE generator, minimizer seed 0, `max_oracle_calls` 2000 (plain LBS with `strict_budget`), every selector cold. Configs cloned from the proteins40 tree by `scripts/gen_selector_dataset_configs.py`.

New here: `generator_cache` (`lab/data/cache/generator_outputs`). The deterministic DCE output is computed once per instance by the first config of each dataset (plain LBS) and replayed for the others, with its oracle calls added back, so every minimizer starts from the identical counterfactual and the OracleCalls metric keeps its meaning; Runtime no longer includes the generator on a replay.

Methods: plain LBS, the thesis selector v1, v2e (the best v2 variant on PROTEINS) and v3b (v2e plus the selector changes picked offline: standardised sweep policy, flip target, tabu memory). Frozen and pretrained arms come afterwards.

```
nohup ~/miniconda3/envs/GRTL/bin/python scripts/run_revision_queue.py --list docs/selector/MULTI40_EXECUTION_ORDER.md --workers 1 --gpu --timeout 21600 > lab/output/queue_multi40.log 2>&1 &
```

- [x] lab/config/generate_minimize/aids40/dce/dce-lcls/generate_minimize0.jsonc
- [x] lab/config/generate_minimize/aids40/dce/dce-lcls-net/generate_minimize0.jsonc
- [x] lab/config/generate_minimize/aids40/dce/dce-lcls-net-v2e/generate_minimize0.jsonc
- [x] lab/config/generate_minimize/aids40/dce/dce-lcls-net-v3b/generate_minimize0.jsonc
- [x] lab/config/generate_minimize/colors340/dce/dce-lcls/generate_minimize0.jsonc
- [x] lab/config/generate_minimize/colors340/dce/dce-lcls-net/generate_minimize0.jsonc
- [x] lab/config/generate_minimize/colors340/dce/dce-lcls-net-v2e/generate_minimize0.jsonc
- [x] lab/config/generate_minimize/colors340/dce/dce-lcls-net-v3b/generate_minimize0.jsonc
