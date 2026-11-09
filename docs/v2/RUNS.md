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

## Last: RHC at equal budget, then COLORS-3

After the fleet, for the expansion of the paper: RHC capped at bls_v1-F's oracle calls in each cell (`dcm-rhc-b`, budgets in `docs/v2/RHC_BUDGET.json`, from the fleet's folds 0 to 8), 210 configs that reuse the cached DCEM seeds:

```bash
launchers/slurm_submit.sh -p normal -m 10 docs/v2/LAST_RHC_B.md
```

### COLORS-3

Its v2 oracle (`OracleTorch-840f1238`) is at chance level (9.8 percent on fold 0): every COLORS-3 cell, the selector's included, has correctness 0. Its oracle is retrained first, then the fleet of `colors-3-s20` and `colors-3-m70-s20` runs after everything else (its configs exist; its queues are generated then).

## Phase 0 (done, for the record)

Fold 0 of the first paper's cells with DCE, DCEM and OFS (`SONDEO_A`, `SONDEO_B`) and of the selector's edge-only arms with DCE (`SONDEO_SEL_A`, `SONDEO_SEL_B`, 8 of 23 cells complete). Findings: `scripts/v2_tables.py --pairs`. Paper 1's BLS stalls when the seed is a counterfactual through its node features (ENZYMES, BBBP), BLS-F does not; DCEM with BLS-F matches DCE at a fraction of the calls; the edge-only selector predicts well (removal AUC 0.85 to 0.97) but does not lower the cost, which is why the fleet uses the F arms.
