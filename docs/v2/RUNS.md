# Protocol v2 on the server: the CPU experiments

Everything the selector paper needs, and the first paper's and thesis' comparisons, rerun on protocol v2 (no padding, `docs/v2/README.md`), so the two can be compared on equal terms. Three phases on the `normal` partition; this page gives the first, and the next two are generated from its hours.

| Phase | What | Configs |
|---|---|---|
| 0, sondeo (this page) | Fold 0 of every first-paper cell (`SONDEO_A`, `SONDEO_B`) and of every selector cell (`SONDEO_SEL_A`, pretraining, `SONDEO_SEL_B`) | 25 + 127 + 92 + 23, one pretraining job |
| 1, selector fleet | The 966 selector configs (23 cells, 10 folds; bls_v0, bls_v1, selector_v1, the untrained control, the collections of folds 1 and 2) minus phase 0 | about 850 |
| 2, first paper and thesis | Level B, the main tables at 10 folds: DCE, DCEM and OFS with the generator alone, BLS, OBS and BLS-F. Level C (revision and ablations: PI4 budgets, PI5, var-1 to var-4, LBS seeds, RHC, DFS): fold 0 only, already in phase 0; their other folds run later | about 900 |

Decisions of 6 November: the selector's bls_v0 and the first paper's BLS (`dce-lcls`) both run, although they are almost the same search on seven datasets (different budgets); BLS-F's place in the selector paper waits for the local comparison on the narrative pairs (`docs/narratives/MINIMIZERS.md`).

The dataset caches, the oracles (retrained without padding and re-saved with CPU tensors) and the DCEM medoids come committed, so no warm job is needed. The generator outputs are built on the server (shared per fold, one seed for every minimizer).

Rule: on the server only Slurm jobs compute. Everything below only syncs git, submits, or reads logs.

## Phase 0, the sondeo

**Step 1. Sync.**

```bash
cd ~/projects/GRETEL && squeue -u $USER && launchers/server_sync.sh && conda activate GRTL
```

**Step 2. Submit both halves at once** (about 10 jobs at a time in total):

```bash
# first paper: the generators alone, then every minimizer from their seeds (B starts when A has ended)
launchers/slurm_submit.sh -p normal -m 5 -c docs/v2/SONDEO_A.md docs/v2/SONDEO_B.md

# selector: bls_v0, bls_v1, control on fold 0 and the collections of fold 1, then the pretraining of the -f1 checkpoints, then selector_v1 on fold 0
launchers/slurm_selector_arm.sh -p normal -m 5 docs/v2/SONDEO_SEL_A.md docs/v2/SONDEO_SEL_B.md --v2 \
    proteins proteins-m70 synthie synthie-m70 bbbp bbbp-m70 aids aids-m70 colors-3-s20 colors-3-m70-s20 bzr bzr-m70 \
    enzymes enzymes-m70 tri-40 tri-40-gcn tri-40-gcn-m70 tri-40n tri-40n-gcn tri-40n-gcn-m70 tri-80 tri-80-gcn tri-80-gcn-m70
```

The second command chains the three selector stages by Slurm dependencies, so nothing else is needed until everything ends. The pretraining job writes the checkpoints to `models/pretrained/v2/` of this clone, where selector_v1 reads them.

**Step 3. While it runs:**

```bash
squeue -u $USER
grep -lE "exit status [1-9]|Traceback" output/logs/v2-*/*/fold*_*.out     # failed configs, if any
tail -1 output/slurm/pretrain_<job>.out     # built 23 checkpoint(s); collections still missing: the fold-2 ones (expected)
```

A queue that ends with failures is submitted again as it is (`launchers/slurm_submit.sh -p normal -m 5 docs/v2/<queue>.md`): finished configs are skipped.

**Step 4. Back home**, on the local machine, when `squeue -u $USER` is empty:

```bash
make server-import
git add lab/output/results/v2-* lab/output/queue_logs lab/output/selector_logs models/pretrained/v2
git commit -m "Protocol v2 sondeo: fold 0 of the first paper's and the selector's cells" && git push
```

Then the analysis (Claude): the first paper's conclusions on fold 0 without padding, the selector checks of the 3 November decisions (bls_v1 at least as good as bls_v0 on bbbp and aids with margin, selector_v1 winning again on tri-80-gcn, selector_v1 against the untrained control), and the hours per config, from which the batches of phases 1 and 2 are generated (`scripts/gen_v2_queues.py`).

## Phases 1 and 2

To be generated after phase 0, in batches of about two days each, in the order of the v1 fleet: the collections of folds 1 and 2 first, the pretraining of every checkpoint, and selector_v1 last. Pending for later: the other nine folds of level C.
