# Rerun of the thesis and paper-revision experiments

Results from before the BLS selector were kept as metrics only and are no longer in the repository (`docs/reproducing-experiments.md`, section 4). This folder reruns every experiment of the LBS thesis (`docs/thesis-experiments.md`) and of the paper revision (`docs/revision/REVISION_EXPERIMENTS.md`) with the committed caches, so that every result has its per-instance counterfactual dumps and every clone gets the same numbers. The old selector (thesis section 5.5 and the `dcm-lcls-net-trainable` row of the final results) is left out: the new selector (`docs/selector/`) replaces it.

180 cells (dataset, generator, minimizer), 10 folds each, 1800 configs, all run on the Slurm server (`docs/selector/SERVER_RUNS.md` for the setup). Everything here is produced by `scripts/gen_rerun_queues.py`, which also turned on the generator cache in these configs (`generator_cache`: each generator runs once per fold and every minimizer starts from the same counterfactual, as the paper's decoupled protocol states; the generator's oracle calls are added back on every reuse). `queues.md` lists every queue.

| Phase | What |
|---|---|
| A | the generator alone (`*-dummy`): fills the generator cache and trains the RSGG GANs, one per fold |
| B | thesis main results: BLS and OBS-min (PI1 to PI3, DCEM tables) |
| C | thesis ablations and baselines: budget (PI4), attributes (PI5), neighbourhood (PI6), `ofs-obs` and `dfs-dbs` of the final results |
| D | paper revision only: DBS, RHC, DFS, Tree-Cycles 300 (E1), LBS seeds (E1b), strategy order (E3) |

**Phase A of a cell must be over before its other runs start.** A minimizer run that finds no stored generator output computes it itself, and OFS and DFS do not give the same counterfactual twice, so two runs computing the same one at the same time could start from different counterfactuals.

## 1. The light arm: one command

Every cell of Tree-Cycles (`tcr-500-28`, `tcr-tco-300`) and ASD: exact or rule-based oracles and few, small instances, 10 s to 2 min per fold on a laptop. The Tree-Cycles generators were fixed on 2026-10-30 (seeded, and generated once per dataset object: before, every process drew different graphs and held each instance twice under the same id, so the evaluator ran each test instance twice and the generator cache replayed another instance's counterfactual) and their dataset caches were regenerated and committed with the fix, so Tree-Cycles results are only valid from that commit on. 660 configs, about 4 hours at 10 at a time (most of it is each job's start-up). Two arrays, the second starting when the first has ended, so phase A comes first:

```bash
launchers/slurm_submit.sh -p normal -m 10 -c docs/rerun/light-1-generators.md docs/rerun/light-2-minimizers.md
```

## 2. Caches built on the server

The ENZYMES dataset and oracle, and the DCM medoids of PROTEINS, BZR and ENZYMES, are not committed yet: a DCM takes from minutes to many hours (BBBP took 14 h on a laptop). BZR's dataset and oracle come out of the selector's local preliminary (`docs/selector/PLAN_PAPER_SELECTOR.md`, section 4.1) and are committed from there, so submit the bzr job only after that sync, when it only has the DCM left to build. One job each, any time:

```bash
G=lab/config/generate_minimize
for ds in bzr enzymes proteins; do sbatch -p normal launchers/slurm_warm.sh $G/$ds/dcm/dcm-dummy/generate_minimize0.jsonc; done
cat output/slurm/warm_<job>.out          # lists what each job loaded or created
```

Once they have finished, bring the new caches back and commit them (the dataset caches hold the folds, so these must be the only copies anyone uses), then `launchers/server_sync.sh` on the server:

```bash
S=kalifano:projects/GRETEL
rsync -av $S/lab/data/cache/datasets/ lab/data/cache/datasets/
rsync -av $S/lab/data/cache/oracles/  lab/data/cache/oracles/
rsync -av --include='*/' --include='DCM-*' --exclude='*' $S/lab/data/cache/explainers/ lab/data/cache/explainers/
git status --short lab/data/cache       # only the new BZR, ENZYMES and PROTEINS files
```

The queues marked *Prep* in `queues.md` need these caches; submit them only after this step.

## 3. The heavy queues

`Q01` to `Q39`, at most 30 configs each, one queue at a time (`Q01` to `Q07` are phase A, and the cells needing the step 2 caches come last in each phase):

```bash
launchers/slurm_submit.sh -p normal -m 10 docs/rerun/Q01.md
squeue -u $USER                      # submit the next queue once this shows nothing
```

A queue that ends with failed configs can be submitted again as it is: `slurm_submit.sh` skips the configs that have results and the ones still queued or running.

## 4. Bringing the results back

With nothing of the batch running (a running config has counterfactual dumps but no `results_<fold>_1.json` yet, and must not be committed), on the local machine:

```bash
ssh -fN kalifano                     # one password for the rsyncs below
S=kalifano:projects/GRETEL
rsync -av $S/lab/output/results/ lab/output/results/
rsync -av --include='*/' --include='GAN-*' --exclude='*' $S/data/ data/
rsync -av $S/output/logs/ lab/output/server_logs/
python scripts/collect_logs.py lab/output/server_logs
git add lab/output/results lab/output/queue_logs data && git commit
```

The generator cache (`lab/data/cache/generator_outputs/`) stays on the server, since it holds one padded graph per test instance and generator. Keep it there until every queue is done: a config rerun after it is deleted recomputes its generator output, which for OFS and DFS may differ from the one the other minimizers started from.
