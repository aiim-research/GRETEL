# Narrative paper, batch 00: the frozen graph pairs (CPU, 3 configs, a few hours)

One GRETEL run per dataset, fold 0, DCE + LBS, no LLM: every instance gives its minimized, seed and random-subset pairs with every generation prompt (`docs/narratives/RUNS.md`, step 1). `launchers/slurm_submit.sh -p normal -m 3 docs/narratives/NARR_B00.md`.

- [ ] lab/config/narratives/pairs/tcr64/generate_minimize0.jsonc
- [ ] lab/config/narratives/pairs/bbbp/generate_minimize0.jsonc
- [ ] lab/config/narratives/pairs/aids/generate_minimize0.jsonc
