# Narrative paper, the model matrix on shared GPUs (Gianrico's account)

Plan C, 16 November 2026. Everything runs locally, from your account, over the frozen pairs that are already in `dev`. Overview: `RUNS.md`. Why and what changes for the paper: `EXPERIMENTS.md`, section 10.

**What was wrong.** Every task waited for the same amount of free GPU memory (70 GB, then 90 GiB), whatever its model, because vLLM takes a fraction of the whole card and refuses to start if that fraction is not free. The 2B narrator needs 12 GiB and was waiting for 90.

**What changes.** Each model waits only for its own memory, and the models are split in three memory classes that are queued apart, so what fits runs now and the rest takes the memory when it appears:

| Class | Models | GiB each | Notes |
|---|---|---|---|
| `s` | narrators Qwen3.5 2B, 4B and 9B, judge GPT-OSS-20B | 12, 18, 28, 22 | as before (bf16, GPT-OSS as stored) |
| `m` | narrators Qwen3.5-27B and Gemma-4-31B-it, judge GLM-4.7-Flash | 42, 46, 42 | loaded in FP8 from the same weights you have (vLLM quantizes them while loading: nothing to download, the card never holds the bf16 model). In bf16 they would need 65 to 73 GiB |
| `l` | judge GPT-OSS-120B | 70 | no smaller version exists. One job waits for the memory and then judges everything with one load |

A task is one model over the three datasets (Tree-Cycles, then BBBP, then AIDS), loaded once: a model that had to wait for memory keeps it until it has finished everywhere. Every command below can be launched again: it only does what is missing.

Rule of the server: only Slurm jobs compute. The commands below only use git, `sbatch`, `squeue`, or read JSON. Run them from the repository root with `conda activate GRTL`. In every new shell, set first:

```bash
cd ~/projects/GRETEL-dev && conda activate GRTL
R=lab/output/results/narr-local
RUNS="--run-root $R/tcr64 --run-root $R/bbbp --run-root $R/aids"
PAIRS="--source lab/output/results/narr-tcr64_dce_lclsf --source lab/output/results/narr-bbbp_dce_lclsf --source lab/output/results/narr-aids_dce_lclsf"
```

**Step 1. Stop the current queue and update the code.** First keep what the waiting jobs learned: the free memory they saw on the GPU, one sample every five minutes. Send Rodrigo the line this prints, it tells us which budgets are realistic:

```bash
grep -ho "[0-9]* MiB free" lab/output/logs/slurm_*.out | awk '{print $1}' | sort -n | awk '{a[NR]=$1} END {i=int(NR*0.9); if (i<1) i=1; if (NR) print "samples", NR, "min", a[1], "median", a[int((NR+1)/2)], "p90", a[i], "max", a[NR]}'
ls lab/output/results/narr-matrix/*/generated/*/generation.json 2>/dev/null     # narrators that finished (none expected)
```

If the second command lists files, tell Rodrigo before going on. Then cancel the queued matrix jobs (they still wait for 90 GiB), put your local edits aside and update:

```bash
squeue -u $USER -o "%.16i %.14j %.9T %.11M %R"                                   # what is queued
squeue -u $USER -h -o "%F %j" | awk '$2 ~ /^pm-/ {print $1}' | sort -u | xargs -r scancel
git stash push -m "my matrix edits, 14 November"                                 # kept: git stash list
launchers/server_sync.sh
ls lab/config/narratives/matrix_local.json                                       # the new configuration
```

The old runs under `narr-matrix` stay where they are, nothing reads them. The environment `GRTL-llm` (vLLM 0.33.0) and the weights are the ones you have. Check that the eight models are in the cache (GPT-OSS-120B is the only one your pilots did not use):

```bash
du -sh ~/hf_cache/hub/models--* | sort -h
sbatch -p normal scripts/slurm_hf_download.sh lab/config/narratives/matrix_local.json   # only if one is missing: it skips the complete ones
```

**Step 2. Load test** (`fit`). Before anything is frozen, each model of classes `s` and `m` is loaded on the real GPU exactly as the matrix will load it, with its budget, and answers the longest prompt of the pairs. A wrong budget, or a model that FP8 does not support, shows here and costs minutes. One job per model, each waiting for its own memory:

```bash
python scripts/paper_probe_matrix.py fit --max 3 --tier s --tier m --config lab/config/narratives/matrix_local.json $PAIRS
python scripts/paper_probe_matrix.py fit --config lab/config/narratives/matrix_local_fp8_control.json --model qwen3.5-9b-fp8 $PAIRS
squeue -u $USER -o "%.16i %.14j %.9T %.11M %R"
```

The small models end within the hour. The three of class `m` need 43 to 47 GiB free and take longer to load. Results:

```bash
grep -H '"status"\|"error"\|"load_seconds"' $R/fit/*.json | grep -v config.json
grep -l "FIT" lab/output/logs/slurm_*.out | xargs grep -h "FIT \|Model loading took\|Available KV cache memory\|Maximum concurrency"
```

Send Rodrigo the output of the second command (vLLM's own numbers: what the weights took, what was left for the cache). What to do with each result:

- `FIT OK` for the four models of class `s`: go on with steps 3 and 4. Do not wait for class `m`.
- `FIT OK` for the three of class `m`: class `m` can be launched (step 5).
- `FIT FAILED` with "is less than desired GPU memory utilization": another user took the memory between the check and the load. Run the test of that model again (`fit --model <id>` with the same other arguments).
- `FIT FAILED` with "No available memory for the cache blocks", or a message about the maximum number of tokens in the KV cache: the budget is short. Run it again with more, e.g. `--memory gemma-4-31b-it-fp8=50`, and tell Rodrigo the value that passes.
- `FIT FAILED` with anything else (in particular an error about quantization or FP8): send Rodrigo the last 30 lines of that job's log. He will push a changed configuration for that model, which enters the runs with `extend` (step 7). Classes that passed go on meanwhile.
- A test of class `m` that is still waiting after two days: tell Rodrigo, with the "MiB free" lines of its log. It means that even 43 GiB are rare.

**Step 3. Freeze the three runs and the control** (CPU jobs, minutes each): each copies the frozen pairs of one dataset into its run, with their hashes.

```bash
for d in tcr64 bbbp aids; do
  sbatch -p normal scripts/slurm_matrix_prepare.sh lab/config/narratives/matrix_local.json $R/$d lab/output/results/narr-${d}_dce_lclsf
done
sbatch -p normal scripts/slurm_matrix_prepare.sh lab/config/narratives/matrix_local_fp8_control.json $R/fp8-control-tcr64 lab/output/results/narr-tcr64_dce_lclsf
tail -qn1 lab/output/logs/matrix_prepare_*.out      # Matrix: .../narr-local/<run>, four times
```

**Step 4. Class `s` on Tree-Cycles alone: the sanity check.** The three small narrators, then their evaluators (GPT-OSS-20B on the three probes and the sweep with one load, and each narrator's self-evaluation). It is short and tells us that the whole pipeline works and how long things take, before the long jobs start.

```bash
python scripts/paper_probe_matrix.py fleet --max 3 --tier s --run-root $R/tcr64
```

`fleet` prints the two arrays (`pm-gen-s`, then `pm-eva-s`, which starts when the narrators' array has ended) and the task of each index. At most `--max` tasks of an array run at a time, the smallest models first. While it runs, and when it ends:

```bash
squeue -u $USER -o "%.16i %.14j %.9T %.11M %R"
python scripts/paper_probe_matrix.py summary --run-root $R/tcr64        # reads JSON only: narrators done, cells complete, error, pending
cut -d, -f1-6 $R/tcr64/matrix-summary.csv
sacct -u $USER -S 2026-11-16 --format=JobID%18,JobName%12,State,Elapsed | grep "pm-"
```

When `summary` shows the three narrators done and their cells with `gpt-oss-20b`, `gpt-oss-20b-sweep` and `self` complete, send Rodrigo the first six columns of the CSV (the parse rates per probe) and the `sacct` lines, and wait for his go before step 5. If the prompts or the parsing were broken, this is where we see it.

**Step 5. Class `s` on the three datasets, and class `m`.** After Rodrigo's go, and for class `m` after its three load tests passed:

```bash
python scripts/paper_probe_matrix.py fleet --max 3 --tier s $RUNS
python scripts/paper_probe_matrix.py fleet --max 2 --tier m $RUNS
```

The first skips Tree-Cycles (done) and does BBBP and AIDS. The second narrates with Qwen3.5-27B and Gemma-4-31B-it, then runs GLM-4.7-Flash over every narrator that is done, GPT-OSS-20B over the two new narrators and their self-evaluations. A class `m` task waits until its GPU has 43 to 47 GiB free, for up to 8 days, and its job can last 10 days. If you know which node has the freest H200, add `--node <node>` (e.g. `--node compute-2-3`). Without it Slurm chooses, and the task waits on the GPU it got.

**Step 6. Class `l`: GPT-OSS-120B.** Launch it when the five narrators are done in the three runs:

```bash
for d in tcr64 bbbp aids; do python scripts/paper_probe_matrix.py summary --run-root $R/$d | grep Narrators; done
python scripts/paper_probe_matrix.py fleet --max 3 --tier l $RUNS
```

One task waits for 71 GiB free (up to 8 days) and then judges the five narrators on the three datasets and the sweep without unloading. The same command also queues GPT-OSS-20B and GLM-4.7-Flash again if any of their cells is still missing, so it doubles as the final sweep of what was left. Two things that help:

- It is safe to launch it once per node (`--node <node>`, one command per node with an H200): the first task that finds the memory takes the work, a second one that starts later shares the cells that are left, and none repeats a cell.
- If class `m` is stuck for more than three days, launch class `l` anyway: it judges the three small narrators now, and the same command later judges the rest.

The FP8 control can go at any time after the load test of `qwen3.5-9b-fp8` passed (one narrator and one judge, class `s` sizes):

```bash
python scripts/paper_probe_matrix.py fleet --max 2 --run-root $R/fp8-control-tcr64
```

**Step 7. When something fails, or something changes.** Launching a class again is always the same `fleet` command, once the previous jobs of that class have ended (`squeue` shows no `pm-gen-<class>` nor `pm-eva-<class>`). It queues only the models with work left. What you may see:

| Symptom | Meaning | What to do |
|---|---|---|
| A task ends with exit status 3, its log says "never ... MiB free" | The memory never appeared in 8 days | Launch the class again, and tell Rodrigo |
| Log: "is less than desired GPU memory utilization" | Another user took the memory between the check and the load | Launch the class again |
| Log: "No available memory for the cache blocks" | The model's budget is short | Launch again with `--memory <id>=<GiB>`, and tell Rodrigo |
| `summary` shows `error` cells | Read `$R/<dataset>/probes/<narrator>/<judge>/error.json` | Launch the class again: it retries them. If it repeats, send the file |
| A job hits the 10-day limit | The cells already complete are kept, the one in progress starts again | Launch the class again |
| Log: "nothing to do now" and the task ends at once | Its narrators have not finished yet, or everything is done | Nothing. A later launch picks it up |

If Rodrigo pushes a new or changed model (e.g. after a failed load test), add it to the prepared runs. Nothing that is frozen or done changes:

```bash
launchers/server_sync.sh
python scripts/paper_probe_matrix.py extend --config lab/config/narratives/matrix_local.json $RUNS
```

**Step 8. The results to `dev`**, after the sanity check and then whenever a class ends:

```bash
git switch -c narr-results                       # the first time; then git switch narr-results
git add lab/output/results/narr-local && git commit -m "Narrative paper: local matrix, <what ended>"
git push fork narr-results
```

Then open (or update) a pull request from `GianFor:narr-results` into `aiim-research:dev`.

**What to send Rodrigo, in order:** the free-memory line of step 1, the numbers of the load test (step 2), the Tree-Cycles summary and times (step 4), and anything that says FAILED or error.
