# Protocol v2 on the server: the CPU experiments

Everything the selector paper needs, and the first paper's and thesis' comparisons, rerun on protocol v2 (no padding, `docs/v2/README.md`), so the two can be compared on equal terms. Three phases on the `normal` partition; this page gives the first, and the next two are generated from its hours.

| Phase | What | Configs |
|---|---|---|
| 0, sondeo (this page) | Fold 0 of every first-paper cell (`SONDEO_A`, `SONDEO_B`) and of every selector cell (`SONDEO_SEL_A`, pretraining, `SONDEO_SEL_B`) | 25 + 127 + 92 + 23, one pretraining job |
| 0-F, sondeo of the F arms (this page) | The same selector stages with BLS-F's substitutions (`SONDEO_SEL_F_A`, pretraining `--features`, `SONDEO_SEL_F_B`) | 92 + 23, one pretraining job |
| 1, selector fleet | The selector arms with substitutions at 10 folds (BLS-F, bls_v1-F, selector_v1-F, the untrained control-F, the collections-F of folds 1 and 2) and bls_v0, minus phases 0 and 0-F. The edge and attribute arms (bls_v1, selector_v1, control) stay at fold 0 as an ablation | to be counted from the hours |
| 2, first paper and thesis (generated, below) | Level B, the main tables at 10 folds: DCE, DCEM (with `fallback_scan`) and OFS with the generator alone, BLS, OBS, RHC (moved from level C on 2026-11-06: the selector paper compares against OBS and hill climbing too) and BLS-F. Level C (revision and ablations: PI4 budgets, PI5, var-1 to var-4, LBS seeds, DFS): fold 0 only, already in phase 0; their other folds run later | about 940 |

Decisions of 6 November: the selector's bls_v0 and the first paper's BLS (`dce-lcls`) both run, although they are almost the same search on seven datasets (different budgets); BLS-F's place in the selector paper waits for the local comparison on the narrative pairs (`docs/narratives/MINIMIZERS.md`). Decision of the same day, after that comparison: BLS-F's substitutions enter the selector's skeleton (`feature_substitutions`, `<arm>-f`): proposed uniformly, the selector still scoring edges only, with their own collections (`collect2f`) and checkpoints (`sel2f-pre`). Phase 0 ends as it is; its edge and attribute arms become the ablation.

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

**Step 2-F. The arms with substitutions. ON HOLD, do not launch:** on 2026-11-06 DCEM was proposed as the base generator from phase 1 on, so the F arms (built here on DCE) are rebuilt after phase 0 ends. The command, for reference:

```bash
launchers/slurm_selector_arm.sh -p normal -m 5 docs/v2/SONDEO_SEL_F_A.md docs/v2/SONDEO_SEL_F_B.md --v2 --features \
    proteins proteins-m70 synthie synthie-m70 bbbp bbbp-m70 aids aids-m70 colors-3-s20 colors-3-m70-s20 bzr bzr-m70 \
    enzymes enzymes-m70 tri-40 tri-40-gcn tri-40-gcn-m70 tri-40n tri-40n-gcn tri-40n-gcn-m70 tri-80 tri-80-gcn tri-80-gcn-m70
```

Same chain: BLS-F, bls_v1-F and the control-F on fold 0 with the collections-F of fold 1, then the `sel2f-pre` checkpoints, then selector_v1-F on fold 0.

**Step 3. While it runs:**

```bash
squeue -u $USER
grep -lE "exit status [1-9]|Traceback" output/logs/v2-*/*/fold*_*.out     # failed configs, if any
tail -1 output/slurm/pretrain_<job>.out     # built 23 checkpoint(s); collections still missing: the fold-2 ones (expected), for each of the two pretraining jobs
```

A queue that ends with failures is submitted again as it is (`launchers/slurm_submit.sh -p normal -m 5 docs/v2/<queue>.md`): finished configs are skipped.

**Step 4. Back home**, on the local machine, when `squeue -u $USER` is empty:

```bash
make server-import
git add lab/output/results/v2-* lab/output/queue_logs lab/output/selector_logs models/pretrained/v2
git commit -m "Protocol v2 sondeo: fold 0 of the first paper's and the selector's cells" && git push
```

Then the analysis (Claude): the first paper's conclusions on fold 0 without padding, the selector checks of the 3 November decisions (bls_v1 at least as good as bls_v0 on bbbp and aids with margin, selector_v1 winning again on tri-80-gcn, selector_v1 against the untrained control), the same checks on the F arms (bls_v1-F against BLS-F, selector_v1-F against bls_v1-F and the control-F), and the hours per config, from which the batches of phases 1 and 2 are generated (`scripts/gen_v2_queues.py`).

## Phase 2, the first paper's cells at 10 folds

Generated on 7 November after the phase-0 analysis (`scripts/v2_tables.py --pairs`). It does not depend on the selector, so it runs next to the selector stages of phase 0. Every DCEM config now carries `fallback_scan` (`scripts/gen_v2_configs.py`): with no medoid projecting to a counterfactual, DCEM goes on with DCE's candidates before falling back. On fold 0 it cut the fallback seeds from 42 to 0 of 200 on AIDS, 7 to 1 of 11 on ASD (as DCE) and 2 to 0 of 41 on BZR. Its new seeds go to their own cache path (suffix `-scan`), but its fold-0 results share their scope with the old ones, so those are deleted first (step 2).

| Queue | What | Configs | Estimate at `-m 5` |
|---|---|---|---|
| `PHASE2_A` | Level B, the generators alone (DCE, DCEM, OFS), folds 0 to 9 | 210 (196 pending after step 2) | a few hours |
| `PHASE2_B` | Level B, every minimizer (BLS, BLS-F, OBS, RHC) from the cached seeds, folds 0 to 9 | 690 | about 26 h with A |
| `PHASE2_C` | Level C, the first paper's ablations (var-1 to var-4, LBS seeds, Synthie budgets, no-attribute datasets, DFS), folds 0 to 9 | 540 (486 pending) | about 11 h |

Level C only serves the thesis (paper 1 is published); it goes last and can be cancelled without touching anything else. COLORS-3 is left out of every queue (below).

**Step 1. Sync** (as in phase 0).

**Step 2. Delete DCEM's fold-0 results**, once no DCEM config of phase 0 is running (the BBBP ones of array 543255 were resubmitted without the scan, so cancel them if they still run):

```bash
squeue -u $USER                                # 543255_0 and 543255_1 are bbbp/dcm-lcls and bbbp/dcm-obs (check: sed -n 1,2p output/slurm/543255.list)
scancel 543255_0 543255_1                      # only if they are still there
ls -d lab/output/results/v2-*_dcm_* | grep -v smoke                  # 32 folders, or 30 if the two BBBP ones never wrote
ls -d lab/output/results/v2-*_dcm_* | grep -v smoke | xargs rm -rf
```

**Step 3. Submit the three queues as a chain** (B starts when A has ended, C when B has ended):

```bash
launchers/slurm_submit.sh -p normal -m 5 -c docs/v2/PHASE2_A.md docs/v2/PHASE2_B.md docs/v2/PHASE2_C.md
```

**Step 4. While it runs and when it ends**: as steps 3 and 4 of phase 0 (failed configs, resubmitting the same command, `make server-import` at home). The results of the selector stages of phase 0 come in the same import.

## Phase 1, the selector fleet

To be generated after the analysis of the selector stages of phase 0 (now running: arrays 543259, pretraining 543260, 543266), around DCEM as the base generator: DCEM twins of the selector arms (with and without BLS-F substitutions), DCEM medoids for the tri-* and -m70 cells, then a fold-0 sondeo of the F chain on DCEM (the old step 2-F) before the 10 folds, in batches of about two days: collections first, the pretraining of every checkpoint, selector_v1 last. Decided on 7 November: the selector runs with DCEM only (no DCE subset); its phase-0 DCE arms stay as they are.

## Last: COLORS-3 and RHC everywhere

Decided on 7 November: RHC on every DCE cell that lacks it (about 60 configs, today it exists on ASD, BBBP, Synthie and TCR-TCO-300 only) is generated and launched together with COLORS-3, after everything else.


Decision of 7 November: its v2 oracle (`OracleTorch-840f1238`) is at chance level (9.8 percent on fold 0), so every colors-3 cell of phase 0 has correctness 0. Its oracle is retrained first, then its cells of every phase (fold 0 included) run after everything else. The selector's COLORS-3 cells (`colors-3-s20`, `colors-3-m70-s20`) use another dataset and oracle and are not affected.
