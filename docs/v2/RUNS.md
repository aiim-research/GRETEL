# Protocol v2 on the server: the CPU experiments

Everything the selector paper needs, on protocol v2 (no padding, `docs/v2/README.md`), with DCEM as the only generator (decision of 8 November). The paper shows, on DCEM seeds, other methods (OBS, RHC) < bls_v0 (paper 1's BLS) < bls_v1 (the selector skeleton with BLS-F's feature substitutions) < selector_v1 (the trained selector on bls_v1), with the selector's ablation (the untrained control on bls_v1) and its statistics at 10 folds. Paper 1's main table (DCEM alone, DCEM with BLS, DCEM with OBS on Synthie, BBBP, ENZYMES, BZR, AIDS, PROTEINS) is a subset of the same runs; its ablations and DCE are not rerun.

| Arm | Config variant | Role |
|---|---|---|
| DCEM alone | `dcm-dummy` | the seeds, and paper 1's generator row |
| OBS, RHC | `dcm-obs`, `dcm-rhc` | other minimizers |
| bls_v0 | `dcm-lcls-v0` | paper 1's BLS (its configuration with the common strict budget of 10 000 calls) |
| bls_v1 | `dcm-lcls-v1-f` | the improved BLS: the skeleton with BLS-F's substitutions, random proposals |
| control | `dcm-lcls-sel-v1-untrained-f` | the selector without training on bls_v1 (ablation) |
| selector_v1 | `dcm-lcls-sel-v1-f` | the trained selector on bls_v1 |
| collections | `dcm-lcls-collect2-f` (folds 1 and 2) | the selector's training data: checkpoint `-f1` serves folds 0 and 2 to 9, `-f2` serves fold 1 |

Cells: the 21 selector cells but COLORS-3 (`docs/v2/FLEET_CELLS.txt`): PROTEINS, Synthie, BBBP, AIDS, BZR, ENZYMES, each also with the margin oracle (`-m70`), and the nine triangle cells. COLORS-3 runs last (below).

DCEM uses the fallback scan (`fallback_scan`: with no medoid projecting to a counterfactual, DCE's candidates before falling back), which on fold 0 cut its fallback seeds to DCE's level (AIDS 42 to 0 of 200, ASD 7 to 1 of 11, BZR 2 to 0 of 41).

Rule: on the server only Slurm jobs compute. Everything below only syncs git, submits, cancels, deletes or reads logs.

## Step 0. Cancel everything that runs (start clean)

The phase-2 queues of 7 November (DCE and the first paper's ablations) and any selector job left are dropped. Only jobs of this account are touched (the narrative runs live in Gianrico's account).

```bash
squeue -u $USER
scancel -u $USER -n gretel
scancel -u $USER -n pretrain
scancel -u $USER -n warm
squeue -u $USER                                # must be empty
```

Nothing has to be deleted: the fleet writes to new scopes (`v2-<cell>-sel_dcm_*`). The results of phase 0 and of the cancelled queues stay as they are (they are not used). The DCEM seeds already cached with the fallback scan are reused.

**Stale locks.** A job cancelled while it builds a cache (DCEM medoids, an oracle) leaves its lock behind (`lab/data/cache/explainers/<dataset>/DCM-<hash>.lck` and its claim files), and every later job that needs that cache waits for it to expire: 120 hours (found on 8 November: five warm jobs idle for 4.5 hours, `sstat` AveCPU of seconds). With no job of this account running, delete them before submitting again:

```bash
find lab/data/cache -name "*.lck*" -delete
```

## Step 1. Sync

```bash
cd ~/projects/GRETEL && launchers/server_sync.sh && conda activate GRTL
```

## Step 2. Train the DCEM medoids that do not exist yet (one warm job per cell)

The six main cells have them (committed, the same as paper 1's cells). The `-m70` cells (another oracle) and the triangle cells do not, and building them inside the fleet would train the same medoids in parallel jobs. A cell whose medoids exist ends in a minute.

At most 10 jobs at a time on the server: the 21 cells go as 10 jobs (two or three cells each, run one after the other inside the job), and the fleet waits for all of them.

```bash
T=lab/config/v2/generate_minimize
JW=""
for group in "aids aids-m70 bbbp" "bbbp-m70 bzr bzr-m70" "enzymes enzymes-m70" "proteins proteins-m70" "synthie synthie-m70" \
             "tri-40 tri-40-gcn" "tri-40-gcn-m70 tri-40n" "tri-40n-gcn tri-40n-gcn-m70" "tri-80 tri-80-gcn" "tri-80-gcn-m70"; do
    cfgs=""; for c in $group; do cfgs="$cfgs $T/$c-sel/dcm/dcm-dummy/generate_minimize0.jsonc"; done
    J=$(sbatch --parsable -p normal launchers/slurm_warm.sh $cfgs)
    JW="${JW:+$JW:}$J"
done
echo "warm jobs: $JW"
```

## Step 3. Submit the fleet as one chain

```bash
JA=$(launchers/slurm_submit.sh -p normal -m 10 -a $JW docs/v2/FLEET_A.md | sed -n 's/^JOBS //p')
echo "fleet A: $JA"
launchers/slurm_selector_arm.sh -p normal -m 10 -a $JA docs/v2/FLEET_B.md docs/v2/FLEET_C.md --v2 --features --dcem $(cat docs/v2/FLEET_CELLS.txt)
```

| Stage | What | Configs | Estimate at `-m 10` |
|---|---|---|---|
| warm | DCEM medoids (new for 15 cells) | 10 jobs | up to several hours (the largest datasets) |
| `FLEET_A` | DCEM alone, folds 0 to 9 | 210 | about 2 h |
| `FLEET_B` | collections-F (folds 1, 2), OBS, RHC, bls_v0, bls_v1-F, control-F, folds 0 to 9 | 1092 | about 30 h |
| pretraining | the `sel2fm-pre` checkpoints, 2 per cell | 1 job | about 1 h |
| `FLEET_C` | selector_v1-F, folds 0 to 9 | 210 | about 6 h |

The queues go fold by fold, so fold 0 of every arm comes first and can be checked while the rest runs: about an hour after `FLEET_B` starts, look for failed configs (step 4) before letting it go on. The estimates come from the phase-0 hours with DCE seeds.

## Step 4. While it runs

```bash
squeue -u $USER
tail -3 output/slurm/warm_*.out                                              # each warm job ends with exit status 0
grep -lE "exit status [1-9]|Traceback" output/logs/v2-*-sel/dcm-*/fold*_*.out  # failed configs
tail -1 output/slurm/pretrain_<job>.out                                      # built 42 checkpoint(s)
```

A queue that ends with failures is submitted again as it is (`launchers/slurm_submit.sh -p normal -m 10 docs/v2/<queue>.md`): finished configs are skipped. If the pretraining fails, run it again by hand before resubmitting `FLEET_C`: `sbatch -p normal launchers/slurm_pretrain.sh --v2 --features --dcem $(cat docs/v2/FLEET_CELLS.txt)` (it skips the checkpoints that exist).

## Step 5. Back home

```bash
make server-import
```

Then the analysis (Claude): the ladder per cell, paper 1's table on DCEM against the published one, the selector's statistics (control, AUC, stuck share), and whatever has to be rerun.

## Arm D (10 November): what the paper still needs

The fleet ended on 10 November (`make server-import`, analysis with `scripts/v2_fleet.py`). The nine triangle cells leave the selector paper (decided 10 November: a study of their own, the starting point of the task-arithmetic line); the paper keeps the 12 classic cells (PROTEINS, Synthie, BBBP, AIDS, BZR, ENZYMES, each also with `-m70`). Arm D, on those 12 cells, folds 0 to 9:

| Queue | What | Configs |
|---|---|---|
| `D.md`, first part | the traced twins of bls_v1-F, control-F and selector_v1-F (`<arm>-t`): the same runs (same seed; checked on BZR fold 0, 41 of 41 instances identical) that also write `lab/output/selector_logs/trace_<tag>_<dataset>_f<fold>.jsonl`, the best solution's edge edits, substitutions and feature cost after every accepted move. It explains why a selector that scores edges only lowers the feature cost, and gives the convergence curves (cost against oracle calls) | 360 |
| `D.md`, second part | RHC at bls_v1-F's budget per cell (`dcm-rhc-b`, `docs/v2/RHC_BUDGET.json`) | 120 |
| warm, then `D_COLORS_A/B/C.md` | COLORS-3 (`colors-3-s20`, `colors-3-m70-s20`) with its oracle fixed: its label is a count of green nodes, which the mean pooling saw only through the padding; v2 pools with a sum (`scripts/gen_v2_configs.py`, `sum_pooling`) | 1 job, then 20, 104 and 20 |

### D1. Submit D and the COLORS-3 warm job (10 jobs at most: 9 + 1)

```bash
cd ~/projects/GRETEL && launchers/server_sync.sh && conda activate GRTL
squeue -u $USER                                # nothing of this account should be running
T=lab/config/v2/generate_minimize
JW=$(sbatch --parsable -p normal launchers/slurm_warm.sh $T/colors-3-s20-sel/dcm/dcm-dummy/generate_minimize0.jsonc $T/colors-3-m70-s20-sel/dcm/dcm-dummy/generate_minimize0.jsonc)
JD=$(launchers/slurm_submit.sh -p normal -m 9 docs/v2/D.md | sed -n 's/^JOBS //p')
echo "warm $JW   D $JD"
```

### D2. Check the COLORS-3 oracle when the warm job ends

```bash
tail -20 output/slurm/warm_$JW.out          # both configs "ok", and a line "Test accuracy = ..."
```

The old oracle was at 0.098 (chance, 11 classes). Go on only with an accuracy clearly above it (paper-level oracles are above 0.9; anything under about 0.5 means the fix is not enough: stop and report it).

### D3. The COLORS-3 fleet, after D (it waits for D's array, so the 10-job limit holds)

```bash
JA=$(launchers/slurm_submit.sh -p normal -m 10 -a $JD docs/v2/D_COLORS_A.md | sed -n 's/^JOBS //p')
launchers/slurm_selector_arm.sh -p normal -m 10 -a $JA docs/v2/D_COLORS_B.md docs/v2/D_COLORS_C.md --v2 --features --dcem $(cat docs/v2/D_COLORS_CELLS.txt)
```

If D has already ended and left the queue, drop `-a $JD`. RHC at equal budget on COLORS-3 needs its bls_v1-F calls first: it is generated after this fleet (add its cells to `docs/v2/RHC_BUDGET.json`).

### D4. Home

`make server-import` brings the results and the traces (they live with the selector monitors).

## Arm E (15 November): what the paper still needs

The selector paper is written (`~/projects/phd/3-2026-11-*/article-es/`). Arm E runs every experiment its TODOs still ask for, on the 12 classic cells and on COLORS-3, folds 0 to 9.

| Queue | What | Configs |
|---|---|---|
| `E.md`, first two lines | the AIDS control-F fold 9 (the arm-D fold that did not reproduce) twice with deterministic PyTorch on one CPU thread (`dcm-lcls-sel-v1-untrained-f-d1`, `-d2`, parameter `deterministic`): if the two coincide, the three folds of arm D differed because of multithreaded float sums (the article's note "determinismo") | 2 |
| `E.md` | DBS on the DCEM seeds (`dcm/dcm-dbs`), OFS with obs (`ofs/ofs-obs`, the coupled explainer of Abrate et al., for the global comparison) and DDBS (`dfs/dfs-dbs`, their data-driven explainer), 12 classic cells | 360 |
| `E_COLORS.md` | the same three arms on the two COLORS-3 cells (DBS needs the DCEM seeds of `D_COLORS_A`) | 60 |
| `E_STAB.md` | selector_v1-F with seeds 1 and 2 (`-s1`, `-s2`), the stability table (decided 2026-11-17: the selector only, against bls_v1-F at seed 0) | 240 |
| later | RHC at bls_v1-F's budget on COLORS-3, once its fleet is home (its budget comes from those runs) | 20 |

**DBS and DFS without padding.** Their prior counted edges by node index (Abrate and Bonchi's, meaningful only when node i is the same node in every graph, as in ASD), so v2 left them out. Arm E runs them with the type-pair prior (`prior: "type_pair"`, `src/explainer/future/utils/data_driven_priors.py`): the edges of the fold's training graphs counted by the types of their endpoints, per class and divided by the number of graphs of each side. The node type per dataset is `NODE_TYPES` in `scripts/gen_v2_configs.py`: the one-hot label of the TU datasets (atom in AIDS and BZR, secondary-structure element in ENZYMES and PROTEINS), the atomic number in BBBP, the five binary columns in COLORS-3, and 10 k-means clusters of the features in Synthie. With the node index as the type and no normalisation it is exactly the index prior (`tests/data_driven_priors_unit.py`). DBS keeps the fleet's obs settings (`max_oc` 4000, batches of 5), DFS OFS's (`max_oc` 2000, batches of 5).

**OFS.** The `ofs-obs` configs of the selector cells are byte for byte paper 1's (`<dataset>/ofs/ofs-obs`) but for the scope (`v2-<cell>-sel_ofs_obs`), so the OFS seeds already cached on the server (fold 0 of the phase-0 sondeo on AIDS, BBBP, BZR, ENZYMES, Synthie) replay. `E.md` replaces the `E_OFS.md` of the morning (dropped): the same runs under the selector scopes, in all 12 cells.

**Cost.** DCEM costs about 3 calls per seed; OFS and DFS search the oracle per instance (up to 2000 calls, about 3600 with obs on BBBP in the sondeo), and DBS, like obs, spends its 4000 calls whenever an edit cannot be undone. Expect `E.md` to take about a day at 9 jobs, the big datasets (AIDS, BBBP, PROTEINS) and the margin oracle the longest. The two deterministic runs use one thread and take a few hours.

### E0. On the laptop: publish the code

The configs, the type-pair prior and the queues are on `dev` once pushed (`git push origin dev`); the server only syncs.

### E1. Sync and look at what is there

```bash
cd ~/projects/GRETEL && launchers/server_sync.sh && conda activate GRTL
squeue -u $USER                                     # nothing of this account should be running
ls lab/output/results | grep -c "^v2-colors-3.*-sel_dcm_"    # COLORS-3 fleet of arm D: 0 if it never ran
find lab/data/cache -name "*.lck*"                  # stale locks: with nothing running, delete them (step 0)
```

If arm D's COLORS-3 fleet (D1 to D3) is already done or queued, skip the COLORS-3 lines below (warm job, D2, `D_COLORS_*`) and chain `E_COLORS.md` after `E.md` directly.

### E2. Submit E and the COLORS-3 warm job (10 jobs at most: 9 + 1)

```bash
T=lab/config/v2/generate_minimize
JE=$(launchers/slurm_submit.sh -p normal -m 9 docs/v2/E.md | sed -n 's/^JOBS //p')
JW=$(sbatch --parsable -p normal launchers/slurm_warm.sh $T/colors-3-s20-sel/dcm/dcm-dummy/generate_minimize0.jsonc $T/colors-3-m70-s20-sel/dcm/dcm-dummy/generate_minimize0.jsonc)
echo "E $JE   warm $JW"
```

About an hour later, check fold 0 (the queue goes fold by fold) before letting it run on:

```bash
grep -lE "exit status [1-9]|Traceback" output/logs/v2-*-sel/{dcm-dbs,ofs-obs,dfs-dbs}/fold*_*.out 2>/dev/null   # failed configs
grep -h "DBS: " output/logs/v2-aids-sel/dcm-dbs/fold0_*.out | head      # "reduced counterfactual (a -> b)" lines
```

### E3. The COLORS-3 oracle, when the warm job ends

As D2: `tail -20 output/slurm/warm_$JW.out`, both configs "ok" and a test accuracy clearly above chance (0.098). Under about 0.5, stop the COLORS-3 part and report it; the rest of arm E does not need it.

### E4. COLORS-3, after E (it waits for E's array, so the 10-job limit holds)

```bash
JA=$(launchers/slurm_submit.sh -p normal -m 10 -a $JE docs/v2/D_COLORS_A.md | sed -n 's/^JOBS //p')
JC=$(launchers/slurm_selector_arm.sh -p normal -m 10 -a $JA docs/v2/D_COLORS_B.md docs/v2/D_COLORS_C.md --v2 --features --dcem $(cat docs/v2/D_COLORS_CELLS.txt) | sed -n 's/^JOBS //p')
JEC=$(launchers/slurm_submit.sh -p normal -m 10 -a $JC docs/v2/E_COLORS.md | sed -n 's/^JOBS //p')
echo "D_COLORS_A $JA   D_COLORS_C $JC   E_COLORS $JEC"
```

If E has already left the queue, drop `-a $JE`. If the COLORS-3 fleet ran before, only the last line, with `-a $JE`.

### E5. The stability table

After the COLORS-3 queue, so the two never exceed 10 jobs together (`JEC` is the E_COLORS job id printed in E4; if the shell lost it, `squeue -u $USER` shows it):

```bash
cd ~/projects/GRETEL && git pull && conda activate GRTL
JS=$(launchers/slurm_submit.sh -p normal -m 10 -a $JEC docs/v2/E_STAB.md | sed -n 's/^JOBS //p'); echo "E_STAB $JS"
```

It reuses the DCEM seeds and the sel2fm-pre checkpoints of the fleet (no warm job). `v2_paper_tables.py` writes the table as section 5 when the results come home: the selector's total with seeds 0, 1 and 2, and each seed against bls_v1-F with Holm across the 12 cells.

### E6. Home

`make server-import`, then (Claude) `scripts/v2_paper_tables.py` (DBS as an arm, the seeds as section 5) and `scripts/v2_traces.py` (the deterministic repetitions in section 0). For RHC-B on COLORS-3: bls_v1-F's mean minimizer calls of its two cells go into `docs/v2/RHC_BUDGET.json`, then `scripts/gen_v2_configs.py`, `scripts/gen_v2_queues.py` and a short queue of its 20 configs.

**What to check when it arrives.** OFS and DFS are not correct on every instance (0.833 on BBBP fold 0 for OFS): the paper compares costs on the instances every explainer gets right and reports correctness as a row. DBS starts from the same DCEM seeds as obs and edits only edges, like obs, so it sits with the other methods and is compared in structure (paper 1's metric).

**Results (2026-11-17).** `E.md` complete, 362 of 362, no failures. DBS on the DCEM seeds loses to obs in structure in the 12 cells at the same calls (+2 % on ENZYMES to +96 % on Synthie) and to bls_v0 in the 12, so it joins obs among the other minimizers. OFS with obs and DDBS never edit a feature (F = 0) and solve 7 % to 100 % of the instances (about 10 % on ENZYMES, 20 % for DDBS on AIDS); the paper compares minimizers on the DCEM seeds only, so they stay out of the tables (decision of 2026-11-17; their results remain under `v2-<cell>-sel_ofs_obs` and `_dfs_dbs`). The deterministic repetitions refute the float-sum explanation: `-d1` and `-d2` equal each other and the non-deterministic traced rerun `-t` (200/200), and the original fold 9 differs from its first instance with the same node, caches, instance order and code. The original is the outlier, cause unknown, and `v2_paper_tables.py` takes the `-t` rerun for the three folds (`RERUN`).

## Step F (25 November): the stability table; COLORS-3 on hold

Home as of 25 November: E (362), D_COLORS_A/B/C (144) and E_COLORS (60) complete; nothing of `E_STAB.md` yet.

**COLORS-3 is on hold.** Its cells evaluate a fixed sample of 20 instances per fold (the evaluator's `instance_ids`), drawn by `scripts/gen_selector_paper_matrix.py` (`subset_ids`) from the splits of the padded dataset cache (`COLORS-3-b24abe85...`). Protocol v2 runs on another dataset cache (`COLORS-3-64f13394...`, isolated nodes deleted) with other splits, so only 23 of the 200 sampled instances fall in their fold's test set: COLORS-3 v2 explained 23 instances per cell, and its selector collections were drawn the same way. `F_COLORS.md` (RHC at bls_v1-F's budget and the traced twins, 80 configs) is ready but waits for that decision: running it now would only add to the same 23 instances.

| Queue | What | Configs |
|---|---|---|
| `E_STAB.md` | selector_v1-F with seeds 1 and 2 on the 12 classic cells (the stability table) | 240 |
| `F_COLORS.md` (on hold) | COLORS-3: RHC at bls_v1-F's budget (101 and 179 calls) and the traced bls_v1-F, control-F and selector_v1-F | 80 |

### F1. Sync and look at what runs

```bash
cd ~/projects/GRETEL && git pull && conda activate GRTL
squeue -u $USER
```

### F2. Submit the stability table

```bash
JS=$(launchers/slurm_submit.sh -p normal -m 10 docs/v2/E_STAB.md | sed -n 's/^JOBS //p'); echo "E_STAB $JS"
```

If `squeue` showed an array of yours still running, add `-a <its job id>` so this one waits for it (10 jobs at most): the launcher leaves out the configs already queued or running in it, and the ones that already have results. No warm job: the DCEM seeds and the sel2fm-pre checkpoints are on the server.

### F3. While it runs

```bash
squeue -u $USER | head
grep -l "status=FAIL\|Traceback" output/logs/v2-*-sel/dcm-lcls-sel-v1-f-s[12]/fold*_${JS}_*.out 2>/dev/null
```

240 runs of the selector, about a day or two at 10 jobs.

### F4. Home

`make server-import`, then (Claude) `scripts/v2_paper_tables.py` (the stability table as its section 5).

## Phase 0 (done, for the record)

Fold 0 of the first paper's cells with DCE, DCEM and OFS (`SONDEO_A`, `SONDEO_B`) and of the selector's edge-only arms with DCE (`SONDEO_SEL_A`, `SONDEO_SEL_B`, 8 of 23 cells complete). Findings: `scripts/v2_tables.py --pairs`. Paper 1's BLS stalls when the seed is a counterfactual through its node features (ENZYMES, BBBP), BLS-F does not; DCEM with BLS-F matches DCE at a fraction of the calls; the edge-only selector predicts well (removal AUC 0.85 to 0.97) but does not lower the cost, which is why the fleet uses the F arms.
