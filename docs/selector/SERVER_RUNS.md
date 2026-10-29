# Running the selector experiments on the Slurm server

Workflow: nothing is edited on the server. Configs and code change on the local machine, get committed and pushed on `dev-bls-selector`, and the server only updates its clone (`launchers/server_sync.sh`, section 1) and submits jobs. Results and logs come back to the local machine with rsync (section 6) and are committed from there, so every clone has them.

## 1. Update the clone and the conda environment

On the server, inside the repository, update with `launchers/server_sync.sh` instead of `git pull`. The first time, the script is not there yet, so run it straight from the branch:

```bash
git fetch && bash <(git show origin/dev-bls-selector:launchers/server_sync.sh)
```

From then on:

```bash
launchers/server_sync.sh            # git fetch + pull, see below
conda activate GRTL                 # or whatever the environment is called there
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu121
python tools/import_smoke.py        # must end with "OK: no module broke that was not already broken"
python -c "import src.explainer.future.metaheuristic.local_search.local_search_selection_net_v2"
```

Caches and results are tracked in git, and a server that already ran experiments has its own untracked copies at the same paths, which make a plain `git pull` stop with "untracked working tree files would be overwritten by merge". `server_sync.sh` deals with exactly those files before pulling: one identical to the committed version (a result produced on the server and committed from the local machine) is removed, one that differs (a dataset or oracle cache trained on the server, which the committed one replaces) is moved to `~/gretel-untracked-backup/<date-time>/` and listed. Nothing tracked is touched.

The environment the published numbers come from is pinned in `requirements-lock.txt` (Python 3.9, torch 2.5.1 from the cu121 wheel index). `pip install -r` only touches what differs from the lock, so it is quick on an existing environment. If the environment does not exist yet: `conda create -n GRTL python=3.9 -y`, then the same `pip install`. The selector work added no dependency outside the lock.

### One-time cleanup: results from before the cut

From the BLS selector onwards, every result must come from the committed caches (same folds, same oracles). Results the server produced earlier (plain BLS, OBS, DBS, hill climbing, ...) used the server's own caches and are discarded: the other methods are rerun later with the committed ones. Once, before the first submission, move them out of the clone (they are untracked, so git does not see this):

```bash
mkdir -p ~/gretel-legacy/results ~/gretel-legacy/logs
# every result except the final experiment's (proteins-sel_*, made with the committed caches)
find lab/output/results -mindepth 1 -maxdepth 1 ! -name 'proteins-sel_*' -exec mv {} ~/gretel-legacy/results/ \;
# every log except those of the Slurm jobs of the final experiment (gretel_*.out)
find output/logs -mindepth 1 -maxdepth 1 ! -name 'gretel_*.out' -exec mv {} ~/gretel-legacy/logs/ \;
launchers/server_sync.sh
```

`collect_logs.py` (section 6) also reads the logs of jobs submitted before the per-folder layout (`output/logs/gretel_<job>_<task>.out`).

## 2. Find the partition you can use

The old launchers hardcoded partitions that changed over time (`normal`, `cuda`, `isolated`, `xai` appear in their history), which is the usual reason for errors like `invalid partition specified` or `user/account not allowed`. There is no public documentation of the DISIM cluster, so ask Slurm directly:

```bash
sinfo -s                                                       # partitions, their state and node counts
sacctmgr show assoc where user=$USER format=account,partition,qos   # which ones your account may use
scontrol show partition <name> | grep -E "MaxTime|DefaultTime|MaxNodes|State"
srun -p <name> -c 1 --mem=1G hostname                          # a one-second test job
```

Pick a CPU partition you are allowed on (these jobs do not need a GPU: the oracle and the selector are small, and every cached oracle is stored on the CPU). If the partition has a short `DefaultTime`, add `#SBATCH -t <time>` to `launchers/slurm_job.sh` locally and push: plain LBS takes several hours per fold at 10000 calls.

## 3. What the experiment needs from git

The comparison is only valid if every job uses the same data splits and the same oracle. Both come from git (`docs/reproducing-experiments.md`, section 4, lists everything that is tracked and why): the dataset caches with their folds in `lab/data/cache/datasets/`, the oracles re-saved on the CPU in `lab/data/cache/oracles/`, the pretrained selector in `models/pretrained/`, and the raw PROTEINS files in `data/working/PROTEINS/PROTEINS/raw/`, so no compute node has to download anything.

The generator's counterfactuals (`lab/data/cache/generator_outputs/`) are not committed: the first job of each fold computes them and the others reuse them (DCE is deterministic).

## 4. Submit

From the repository root on the server:

```bash
# one job first, to check the setup (the selector's fold 0, about 2 to 3 hours)
launchers/slurm_submit.sh -p <partition> -n 1 docs/selector/SERVER_PROTEINS_FOLDS.md

# once it runs fine: everything still pending in the main queue, at most 10 jobs at a time
launchers/slurm_submit.sh -p <partition> -m 10 docs/selector/SERVER_PROTEINS_FOLDS.md

# later, the control arms (untrained and random selector)
launchers/slurm_submit.sh -p <partition> -m 10 docs/selector/SERVER_PROTEINS_CONTROLS.md
```

`slurm_submit.sh` builds the list of configs that still have no results (`scripts/pending_configs.py`), writes it to `output/slurm/<date-time>.list` (kept as `output/slurm/<arrayjob>.list` once submitted), and submits it as one job array throttled to `-m` concurrent tasks (`--array=0-N%10`): Slurm itself starts the next pending task as soon as a running one ends, so nothing has to be watched or resubmitted by hand. Each task runs one config with 4 CPUs (BLAS and torch capped at 4 threads) and 8 GB (`launchers/slurm_job.sh`). Resubmitting the same queue is safe: finished configs are skipped, and so are the configs still queued or running in an earlier array (read from `squeue` and that array's list), so after a failure or a cancellation, or after the single test job, just submit again.

Logs: one per config, in one folder per dataset, generator and minimizer, `output/logs/<dataset>/<generator>-<minimizer>/fold<f>_<job>_<task>.out` (for example `output/logs/proteins-sel/dce-lcls-sel/fold0_123456_0.out`). Each starts with the config, node and python it ran with and ends with its exit status. What Slurm itself prints before the job opens its log (a failed conda activation, for instance) is in `output/slurm/gretel_<job>_<task>.out`.

With `-c`, each queue file becomes its own array, started when the previous one has ended (`--dependency=afterany`), so several queues that must run in order go out with one command.

Queues: `SERVER_PROTEINS_FOLDS.md` (selector and plain LBS, 10 folds each), `SERVER_PROTEINS_CONTROLS.md` (untrained and random selector arms) and `SERVER_PROTEINS_FALLBACK.md` (the selector with a uniform fallback, for the two instances where it never left its initial counterfactual). The rerun of the thesis and paper-revision experiments has its own 61 queues, order and rules in `docs/rerun/README.md`.

## 5. Watch and cancel

```bash
squeue -u $USER                                         # running and pending tasks (<job>_<i>)
ls output/logs/proteins-sel/*/                          # every log, by generator and minimizer
tail -f output/logs/proteins-sel/dce-lcls-sel/fold0_<job>_0.out
grep -l "exit status 0"            output/logs/*/*/*_<job>_*.out   # finished tasks
grep -lE "exit status [1-9]|Traceback" output/logs/*/*/*_<job>_*.out
cat output/slurm/gretel_<job>_<i>.out                    # if a task has no log under output/logs
sacct -j <job> --format=JobID,State,Elapsed,MaxRSS
scancel <job>                                           # the whole array;  scancel <job>_<i> for one task
```

## 6. Bring the results back and commit them

From the local machine (read-only on the server):

```bash
rsync -av <user>@<server>:<path>/GRETEL/lab/output/results/proteins-sel_* lab/output/results/
rsync -av <user>@<server>:<path>/GRETEL/output/logs/ lab/output/server_logs/
python scripts/collect_logs.py lab/output/server_logs
git add lab/output/results lab/output/queue_logs && git commit
```

`collect_logs.py` keeps, for each config, the last attempt that finished with status 0 and writes it gzipped to `lab/output/queue_logs/<dataset>/<generator>-<minimizer>/fold<f>.log.gz`, where git tracks it and the analysis scripts find it (`lab/output/server_logs/` itself is not tracked). Commit only complete runs: a config whose `results_<fold>_1.json` is in git counts as done on every clone. After the push, `launchers/server_sync.sh` on the server removes its now-identical copies and pulls.

Then analyse with `scripts/analyze_selector_runs.py stats`, `scripts/compare_instances.py` (per-instance GED and calls; `--logs proteins-sel_dce_lcls:0 proteins-sel_dce_lcls-sel:0` adds the calls up to each method's last improvement) and `scripts/selector_trajectories.py`.

## 7. The old launchers

`launchers/m_multi.sh` looped over the configs of one hardcoded directory and ran `sbatch launchers/m_launch.sh future_main.py <config> <run>` for each, with a random pause between submissions; `m_launch.sh` asked for 4 CPUs on a hardcoded partition and wrote every job into the same `output.out`. They submit one independent job per config with no concurrency limit, no skipping of finished configs and no per-job log, and `future_main.py` sets 16 threads per process whatever the job was granted. `slurm_submit.sh` and `slurm_job.sh` replace them for this work.
