# Running the selector experiments on the Slurm server

Workflow: nothing is edited on the server. Configs and code change on the local machine, get committed and pushed on `dev-bls-selector`, and the server only does `git pull` and submits jobs. Results stay on the server under `lab/output/results/` and are copied back for analysis (section 6).

## 1. Update the conda environment

The environment the published numbers come from is pinned in `requirements-lock.txt` (Python 3.9, torch 2.5.1 from the cu121 wheel index). On the server, inside the repository:

```bash
git pull
conda activate GRTL          # or whatever the environment is called there
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu121
python tools/import_smoke.py # must end with "OK: no module broke that was not already broken"
python -c "import src.explainer.future.metaheuristic.local_search.local_search_selection_net_v2"
```

`pip install -r` only touches what differs from the lock, so it is quick on an existing environment. If the environment does not exist yet: `conda create -n GRTL python=3.9 -y`, then the same `pip install`. The selector work added no dependency outside the lock (scipy, scikit-learn and filelock were already there).

## 2. Find the partition you can use

The old launchers hardcoded partitions that changed over time (`normal`, `cuda`, `isolated`, `xai` appear in their history), which is the usual reason for errors like `invalid partition specified` or `user/account not allowed`. There is no public documentation of the DISIM cluster, so ask Slurm directly:

```bash
sinfo -s                                                       # partitions, their state and node counts
sacctmgr show assoc where user=$USER format=account,partition,qos   # which ones your account may use
scontrol show partition <name> | grep -E "MaxTime|DefaultTime|MaxNodes|State"
srun -p <name> -c 1 --mem=1G hostname                          # a one-second test job
```

Pick a CPU partition you are allowed on (these jobs do not need a GPU: the oracle and the selector are small, and the cached oracle is stored on CPU). If the partition has a short `DefaultTime`, add `#SBATCH -t <time>` to `launchers/slurm_job.sh` locally and push: plain LBS takes several hours per fold at 10000 calls.

## 3. What the experiment needs, and why it is in git

The comparison is only valid if every job uses the same data splits and the same oracle as the runs made locally. Both are tiny and are committed despite `.gitignore`:

- `lab/data/cache/datasets/PROTEINS-084c6a1896e8a96ebacf7ff4868f1197`: the PROTEINS dataset with its 10 folds (the folds are drawn at random the first time, so a server that built its own cache would split differently);
- `lab/data/cache/oracles/PROTEINS-.../OracleTorch-79176f00a96310661635222d09206cd6`: the trained GCN oracle, re-saved on CPU so it loads on any node (identical predictions on CPU and GPU, checked);
- `models/pretrained/edge_selector_v3-pre_PROTEINS.pt`: the selector's offline-pretrained ranker;
- `data/working/PROTEINS/PROTEINS/raw/`: the raw TUDataset files, so no compute node has to download anything.

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

`slurm_submit.sh` builds the list of configs that still have no results (`scripts/pending_configs.py`), writes it to `output/slurm/<date-time>.list`, and submits it as one job array throttled to `-m` concurrent tasks (`--array=0-N%10`). Each task runs one config with 4 CPUs and 8 GB (`launchers/slurm_job.sh`) and logs to `output/logs/gretel_<arrayjob>_<task>.out`; the log starts with the config, node and python it ran with. Resubmitting the same queue is safe: finished configs are skipped, so after a failure or a cancellation just submit again.

Queues: `SERVER_PROTEINS_FOLDS.md` (selector and plain LBS, 10 folds each; plain LBS fold 0 reruns there so all numbers come from the same machine) and `SERVER_PROTEINS_CONTROLS.md` (untrained and random selector arms).

## 5. Watch and cancel

```bash
squeue -u $USER                          # running and pending tasks (array tasks show as <job>_<i>)
tail -f output/logs/gretel_<job>_0.out   # one task's log
grep -l "status=OK"   output/logs/gretel_<job>_*.out   # finished tasks
grep -l "status=FAIL\|Traceback" output/logs/gretel_<job>_*.out
sacct -j <job> --format=JobID,State,Elapsed,MaxRSS
scancel <job>                            # the whole array;  scancel <job>_<i> for one task
```

## 6. Bring the results back

From the local machine (read-only on the server):

```bash
rsync -av <user>@<server>:<path>/GRETEL/lab/output/results/proteins-sel_* lab/output/results/
rsync -av <user>@<server>:<path>/GRETEL/output/logs/ lab/output/server_logs/
```

Then analyse locally with `scripts/analyze_selector_runs.py stats`, `scripts/compare_instances.py` (per-instance GED and calls; its `--logs` option takes the job logs for the calls up to each method's last improvement) and `scripts/selector_trajectories.py`.

## 7. The old launchers

`launchers/m_multi.sh` looped over the configs of one hardcoded directory and ran `sbatch launchers/m_launch.sh future_main.py <config> <run>` for each, with a random pause between submissions; `m_launch.sh` asked for 4 CPUs on a hardcoded partition and wrote every job into the same `output.out`. They submit one independent job per config with no concurrency limit, no skipping of finished configs and no per-job log, and `future_main.py` sets 16 threads per process whatever the job was granted. `slurm_submit.sh` and `slurm_job.sh` replace them for this work.
