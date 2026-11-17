# Narrative paper, Rodrigo's account

Since 6 November every GPU run is in Gianrico's account (`RUNS_GIANRICO.md`): his quota holds the weights, which do not fit in ours (64 GB). This account computed the pairs, and runs the CPU work of protocol v2 (`docs/v2/RUNS.md`).

**Plan C (16 November).** Everything local, each model with its own GPU budget, in three memory classes (`EXPERIMENTS.md`, section 10). On this side:

- Read what Gianrico sends and answer: the memory numbers of his finished jobs and Gemma's error (his step 1), the numbers of the load test (if a budget is short or Gemma fails in FP8 too, change `lab/config/narratives/matrix_local.json`, giving a changed model a new id, push, and he runs `extend`), and the Recourse results on Tree-Cycles (his step 4), which need a go before the rest is launched.
- Tree-Cycles is his run of 14 November, reused (`EXPERIMENTS.md`, section 12). Its results are analysed in section 11.
- Ask Francesco and Giovanni for a reservation of one H200 for two days. GPT-OSS-120B needs 71 GiB free at once and that has not been seen in a week. With a reservation it judges everything in one job.
- Decide the primary judge when class `l` has either run or waited its 8 days without memory.

**The pairs (done, 5 November).** For each fold-0 instance of a dataset (an input graph), the counterfactuals the narrators will explain, in three arms: the DCE seed, its BLS-F minimization, and a random subset of the seed's edits of the minimized size (the size control). One GRETEL run per dataset (DCE + BLS-F, no LLM, `lab/config/narratives/pairs/<dataset>/`) wrote one JSON per instance under `lab/output/results/narr-<dataset>_dce_lclsf/.../probe_inputs/`, with both graphs of every arm, the oracle's classes, the true typed edit set (the ground truth of the probes), the renaming permutation, the Recourse budget K, and the exact prompts every narrator will receive (direct, inverse, renamed, identity). `prepare` copies them into each run of the matrix with their hashes. They are in `dev`: Tree-Cycles 64 (100 instances), BBBP (102) and AIDS (100), every arm valid. How they were made, to make them again:

```bash
cd ~/projects/GRETEL && launchers/server_sync.sh && conda activate GRTL
launchers/slurm_submit.sh -p normal -m 3 docs/narratives/NARR_B00.md
for d in tcr64 bbbp aids; do echo $d $(find lab/output/results/narr-${d}_dce_lclsf -name 'cf_*.json' | wc -l); done
```

Then, at home: `rsync` of `lab/output/results/narr-*_dce_lclsf`, commit and push to `dev`.

The GPU environment `GRTL-llm` built in this account on 5 November (`scripts/slurm_setup_llm_env.sh`) stays, for small GPU tests; the weights of the paper's models are not downloaded here.
