# Narrative paper, the model matrix on shared GPUs (Gianrico's account)

Updated 17 November 2026, after your Tree-Cycles run and the memory numbers of its logs. Overview: `RUNS.md`. Why and what it means for the paper: `EXPERIMENTS.md`, sections 10 to 12.

**Where we are.** Your run `narr-matrix/tcr64-20b-h200-longwait` is kept and completed, not repeated: the narratives of Qwen3.5 2B, 4B, 9B and 27B and their Reconstruction and Consistency results (GPT-OSS-20B, GLM-4.7-Flash, self-evaluation) are final. What is left:

| What | Why |
|---|---|
| Recourse on Tree-Cycles, judges only | It failed because NetworkX 3.7 no longer has `nx.random_tree`. Fixed in `dev` (the same trees as before) |
| Gemma-4-31B-it on the three datasets, in bf16 | It needs 91 GiB: 58 of weights and 29 of KV cache for one 40 960-token prompt, and your jobs gave it 84 |
| BBBP and AIDS, everything | Nothing ran there yet |
| GPT-OSS-120B on the three datasets | Whenever a GPU has 71 GiB free |

**How it runs now.** Each model waits only for its own memory, and the models are queued in three memory classes:

| Class | Models | GiB each | GPUs |
|---|---|---|---|
| `s` | narrators Qwen3.5 2B, 4B, 9B, judge GPT-OSS-20B | 12, 18, 28, 22 | any |
| `m` | narrator Qwen3.5-27B, judge GLM-4.7-Flash | 60, 64 | any, in practice an H200 |
| `l` | narrator Gemma-4-31B-it, judge GPT-OSS-120B | 93, 70 | H200 only |

Every model runs as in your run: bf16, and GPT-OSS as stored. The budgets come from your logs (weights, plus the KV cache of one prompt of the maximum length, plus what vLLM needs besides).

A task is one model over the three datasets (Tree-Cycles, BBBP, AIDS), loaded once. Every command can be launched again: it only does what is missing, so Tree-Cycles' finished narrators and cells are skipped. Rule of the server: only Slurm jobs compute. The commands below only use git, `sbatch`, `squeue`, or read JSON. In every new shell:

```bash
cd ~/GRETEL && conda activate GRTL
TCR=lab/output/results/narr-matrix/tcr64-20b-h200-longwait
R=lab/output/results/narr-local
RUNS="--run-root $TCR --run-root $R/bbbp --run-root $R/aids"
PAIRS="--source lab/output/results/narr-tcr64_dce_lclsf --source lab/output/results/narr-bbbp_dce_lclsf --source lab/output/results/narr-aids_dce_lclsf"
```

Do not move or rename `$TCR`: its manifest records the absolute paths of its frozen pairs.

**Step 1. Update the code.** The memory numbers of your logs are in (thank you): they set the budgets above. Your clone has your own changes of 14 November (a local commit and maybe edits): they are kept on a branch and in a stash, and `dev` is moved to the published one. Results and other untracked files stay where they are.

```bash
squeue -u $USER                                  # should be empty: everything of 14 November has ended
git fetch origin
git branch keep-14-november                      # your current state, local commit included
git status --short | grep -v '^??'               # if this lists files: git stash push -m "edits of 14 November"
git checkout -f -B dev origin/dev
git log --oneline -1                             # the latest commit of dev
```

If you already did this, run only `git fetch origin && git checkout -f -B dev origin/dev` again: the budgets changed on the 17th.

If you never warmed the AIDS raw data (the Recourse probe rebuilds the molecules for the oracle), do it once:

```bash
sbatch -p normal --wrap "bash -lc 'conda activate GRTL && python tools/warm_dataset.py lab/config/narratives/pairs/aids/generate_minimize0.jsonc'"
```

**Step 2. Load test** (`fit`). Each model of classes `s` and `m` is loaded on the real GPU exactly as the matrix will load it, with its budget, and answers the longest prompt of the pairs. A budget that is too tight shows here and costs minutes. One job per model, each waiting for its own memory (class `l` is not tested: its test would wait as long as its real job):

```bash
python scripts/paper_probe_matrix.py fit --max 3 --tier s --config lab/config/narratives/matrix_local.json $PAIRS
python scripts/paper_probe_matrix.py fit --max 2 --tier m --node compute-2-3 --config lab/config/narratives/matrix_local.json $PAIRS
squeue -u $USER -o "%.16i %.14j %.9T %.11M %R"
```

Results, when they end:

```bash
grep -H '"status"\|"error"\|"load_seconds"' lab/output/results/narr-local/fit/*.json | grep -v config.json
grep -l "FIT" lab/output/logs/slurm_*.out | xargs grep -h "FIT \|Model loading took\|Available KV cache memory\|Maximum concurrency"
```

Send Rodrigo the second output. What to do with each result:

- `FIT OK` for a model: its class can go (steps 4 to 6). Do not wait for the others.
- `FIT FAILED` with "is less than desired GPU memory utilization": another user took the memory between the check and the load. Run that test again (`fit --model <id>` with the same other arguments).
- `FIT FAILED` with "No available memory for the cache blocks", or "KV cache is needed, which is larger than the available KV cache memory": the budget is short. Run it again with more, e.g. `--memory qwen3.5-27b=66`, and tell Rodrigo the value that passes.
- `FIT FAILED` with anything else: send Rodrigo the last 30 lines of that job's log.
- A test still waiting after two days: tell Rodrigo, with the "MiB free" lines of its log.

**Step 3. Prepare the runs** (CPU jobs, minutes). BBBP and AIDS get new runs, with the frozen pairs and their hashes.

```bash
for d in bbbp aids; do
  sbatch -p normal scripts/slurm_matrix_prepare.sh lab/config/narratives/matrix_local.json $R/$d lab/output/results/narr-${d}_dce_lclsf
done
tail -qn1 lab/output/logs/matrix_prepare_*.out      # Matrix: .../narr-local/<run>, twice
```

Your Tree-Cycles run is brought to the same configuration instead (JSON only, seconds). `extend` adds GPT-OSS-120B and sets the budgets. `reset` moves the failed Recourse results to the archive, so those cells do Recourse again and nothing else:

```bash
python scripts/paper_probe_matrix.py extend --config lab/config/narratives/matrix_local.json --run-root $TCR
python scripts/paper_probe_matrix.py reset --probe recourse --judge gpt-oss-20b --judge gpt-oss-20b-sweep --judge glm-4.7-flash --run-root $TCR
python scripts/paper_probe_matrix.py summary --run-root $TCR
```

The last command should print four narrators done and Gemma missing, and `{'complete': 4, 'error': 4, 'pending': 22}`. The four complete cells are the self-evaluations, which have no Recourse. The four errors are Gemma's failed start of the 16th: they are retried once Gemma has narrated.

**Step 4. Recourse on Tree-Cycles: the sanity check.** It is the only probe that never ran, so it goes first and alone. GPT-OSS-20B judges Recourse for the four narrators and the budget sweep with one load:

```bash
python scripts/paper_probe_matrix.py fleet --max 3 --tier s --run-root $TCR
```

Only one task is queued (`pm-eva-s`), since the narrators are done. When it ends:

```bash
python scripts/paper_probe_matrix.py summary --run-root $TCR
grep recourse $TCR/matrix-summary.csv | cut -d, -f1-6
sacct -u $USER -S 2026-11-17 --format=JobID%18,JobName%12,State,Elapsed | grep "pm-"
```

Send Rodrigo the Recourse lines and the times, and wait for his go. The status counts must show no `oracle_error`.

**Step 5. Class `s` on the three datasets, and class `m`.** After Rodrigo's go, and for class `m` after its two load tests passed:

```bash
python scripts/paper_probe_matrix.py fleet --max 3 --tier s $RUNS
python scripts/paper_probe_matrix.py fleet --max 2 --tier m $RUNS
```

The first narrates BBBP and AIDS with the three small Qwen models and judges them with GPT-OSS-20B. The second narrates BBBP and AIDS with Qwen3.5-27B (Tree-Cycles is done), then GPT-OSS-20B judges it, it judges itself, and GLM-4.7-Flash judges everything it has not judged (Recourse on Tree-Cycles, and BBBP and AIDS). A task waits until its GPU has its budget free, for up to 8 days, and its job can last 10 days. The models of class `m` need 61 and 65 GiB free, which an 80 GB card only has when nobody else uses it, so add `--node compute-2-3` (or another H200 node) to the class `m` command. Without it Slurm chooses, and the task waits on the GPU it got.

**Step 6. Class `l`.** Launch it when the narrators' arrays of step 5 (`pm-gen-s`, `pm-gen-m`) have ended, always on an H200 node:

```bash
python scripts/paper_probe_matrix.py fleet --max 3 --tier l --node compute-2-3 $RUNS
```

Gemma narrates the three datasets when its GPU has 94 GiB free (your jobs found 90 GiB twice on the 16th). Then GPT-OSS-20B, GLM-4.7-Flash and Gemma itself judge its narratives, and GPT-OSS-120B judges everything with one load, when it finds 71 GiB. The same command also queues GPT-OSS-20B and GLM-4.7-Flash again for whatever is still missing, so it doubles as the final sweep. Two things that help:

- It is safe to launch it once per H200 node (`--node <node>`, one command per node): the first task that finds the memory takes the work, one that starts later shares the cells that are left, and none repeats a cell.
- Gemma's narratives are needed before its cells can be judged, but GPT-OSS-120B does not wait for them: it judges the four Qwen narrators at once, and the same command launched later judges Gemma.

If Gemma has not found 94 GiB after a week, tell Rodrigo: the fallback is a lower precision for its KV cache, which halves its 29 GiB.

**Step 7. When something fails, or something changes.** Launching a class again is always the same `fleet` command, once the previous jobs of that class have ended (`squeue` shows no `pm-gen-<class>` nor `pm-eva-<class>`). It queues only the models with work left.

| Symptom | Meaning | What to do |
|---|---|---|
| A task ends with exit status 3, its log says "never ... MiB free" | The memory never appeared in 8 days | Launch the class again, and tell Rodrigo |
| A task ends at once with exit status 4, its log says "queue it on an H200 node" | It got an 80 GB card and its model needs more | Launch the class again with `--node <H200 node>` |
| Log: "is less than desired GPU memory utilization" | Another user took the memory between the check and the load | Launch the class again |
| Log: "No available memory for the cache blocks" | The model's budget is short | Launch again with `--memory <id>=<GiB>`, and tell Rodrigo |
| `summary` shows `error` cells | Read `<run>/probes/<narrator>/<judge>/error.json` | Launch the class again: it retries them. If it repeats, send the file |
| A job hits the 10-day limit | The cells already complete are kept, the one in progress starts again | Launch the class again |
| Log: "nothing to do now" and the task ends at once | Its narrators have not finished yet, or everything is done | Nothing. A later launch picks it up |

If Rodrigo pushes a changed configuration, bring the runs to it. Nothing that is frozen or done changes:

```bash
git fetch origin && git checkout -f -B dev origin/dev
python scripts/paper_probe_matrix.py extend --config lab/config/narratives/matrix_local.json $RUNS
```

**Step 8. The results to `dev`**, after step 4 and then whenever a class ends:

```bash
git switch -c narr-results                       # the first time; then git switch narr-results && git merge dev
git add $TCR $R && git commit -m "Narrative paper: local matrix, <what ended>"
git push fork narr-results
git switch dev
```

Then open (or update) a pull request from `GianFor:narr-results` into `aiim-research:dev`.

**What to send Rodrigo, in order:** the numbers of the load test (step 2), the Recourse lines and times of step 4, and anything that says FAILED or error.
