# Narrative paper: running the experiments on the server

What `docs/narratives/EXPERIMENTS.md` plans, as Slurm jobs on kalifano. Two kinds of work:

- **Pairs pass** (CPU, GRETEL, protocol v2: no padding, isolated nodes deleted, `docs/v2/README.md`): one run per dataset, fold 0, DCE + BLS-F, no LLM. Every instance gives three graph pairs, the arms `minimized` (BLS-F: bond edits and atom substitutions), `seed` (DCE's nearest molecule of the other class, written as edits on the input: bond edits, atom substitutions, atoms deleted) and `random_subset` (the input plus a random subset of the seed's edits of the minimized size), and freezes every generation prompt: direct, inverse, renamed (nodes permuted, permutation saved) and identity (the direct prompt again, sampled with seed + 1). The random subset only gets the direct prompt. Deleted atoms are left out of the counterfactual's text and reported as `nodes_removed`. Configs: `lab/config/narratives/pairs/{tcr64,bbbp,aids}/generate_minimize0.jsonc`; results in `lab/output/results/narr-<dataset>_dce_lclsf`.
- **Model matrix** (GPU, `scripts/paper_probe_matrix.py`): the pairs are frozen into a run, and each model is one Slurm job on one `fast` GPU of the `cuda` partition (`compute-0-3` has four, `compute-2-3` one). Generator jobs write the narratives, judge jobs run the probes over every generator's narratives, and self jobs let each narrator judge its own Consistency inversion test (E6). In the full matrix the jobs of one role run side by side (`"parallel": true`), so the generators, the judges and the self jobs are submitted as three batches; the E0 pilot chains its three jobs. Configs: `lab/config/narratives/matrix.json` (the paper) and `matrix_e0.json` (the pilot: the first 10 pairs of Tree-Cycles and BBBP, Qwen2.5-7B, the primary judge).

Models, all local until there is API access: narrators Qwen2.5-Instruct 3B, 7B, 14B and 32B and Llama-3.1-8B-Instruct in bf16 with vLLM, temperature 0.7, top-p 0.95, 1024 output tokens, seed 0 (identity: seed 1). Judges: GPT-OSS-120B as the primary judge of the three probes (temperature 1, low reasoning effort, as the pilots recommend: greedy decoding looped), the same model for the Recourse budget sweep (`gpt-oss-120b-sweep`), and GLM-4.7-Flash as a local robustness judge on a subset (Reconstruction CTX-off/MODE-dict, Consistency, Recourse). No judge reads narratives of its own family. DeepSeek-V3.2, Gemini 3 Flash and the Gemini 2.5 Flash narrator wait for API access. Exact revisions in the configs.

Probes per narrator and pair (primary judge): Reconstruction under CTX on/off x MODE dict/full on the three arms; Consistency (inversion, renaming, identity) and Recourse with and without explanation at K = the instance's minimized edit count, on the seed and minimized arms. The sweep: K in 1, 2, 3, 5, 8, minimized arm, Tree-Cycles and BBBP.

| Batch | Jobs | Estimate | Content |
|---|---|---|---|
| B00 | 3 CPU configs + 1 download | under 3 h | pairs pass of Tree-Cycles 64, BBBP and AIDS (E1 to E3 inputs); model weights into `~/hf_cache` |
| B01 | 1 CPU + 3 GPU, chained | 1 to 2 h | E0 pilot: Qwen2.5-7B narrates 20 instances, primary judge, self-evaluation |
| B02 | 1 CPU + 5 GPU in parallel | 1 to 3 h | E1 and the E3 narratives: the five narrators over every pair (13,500 generations) |
| B03 | 3 GPU in parallel | 6 to 18 h | E2 and E3 scoring (primary judge, about 33,000 calls), E4 budget sweep (10,000) and the GLM robustness judge (about 21,000) |
| B04 | 5 GPU in parallel | under 2 h | E6 self-evaluation (2,000 calls) |

The estimates come from the call counts and an assumed vLLM throughput on one large GPU, not from measurements: B01 measures them (step 3). Not in any batch yet: E5 with the API judges and the Gemini narrator; E7, E8 and the analyses run offline on the local machine.

Rules: on the server only Slurm jobs compute. `launchers/slurm_submit.sh` and `python scripts/paper_probe_matrix.py submit` only list configs or read the run's JSON and call `sbatch`; `prepare`, which imports the framework, runs as a CPU job (`scripts/slurm_matrix_prepare.sh`). The server's clone works on the `dev` branch (shared with the protocol v2 sondeo, `docs/v2/RUNS.md`). Each GPU job first waits until its GPU has `min_free_mb` (70 GB) free, for up to 6 hours (`max_wait`), and fails with exit status 3 if it never does.

**Step 0. Once: the `dev` branch on the server's clone and the GPUs** (git, environment checks and a one-second Slurm job). Switch only when no job of the clone is queued or running: a task that starts after the switch runs the new branch's code.

```bash
cd ~/projects/GRETEL && squeue -u $USER                 # empty before switching
git fetch && git switch dev && launchers/server_sync.sh
conda activate GRTL && conda list vllm        # the GPU jobs need vLLM in this environment (another one: export GRETEL_ENV=<env>)
srun -p cuda --gres=gpu:fast:1 -c 1 --mem=1G nvidia-smi --query-gpu=name,memory.total --format=csv
```

If the `fast` cards have less than about 100 GB, Qwen2.5-32B and GPT-OSS-120B do not fit at `gpu_memory_utilization` 0.75: say so before B02, and their entries in `matrix.json` get 0.9 or two GPUs (`tensor_parallel_size` 2, `gres` `gpu:fast:2`).

Llama-3.1-8B is gated: accept its license on huggingface.co with the account of the token, then `huggingface-cli login` once (or `export HF_TOKEN=...` before the download job).

**Step 1. Batch 00** (CPU, under 3 hours), the pairs and the weights in parallel:

```bash
cd ~/projects/GRETEL && launchers/server_sync.sh && conda activate GRTL
launchers/slurm_submit.sh -p normal -m 3 docs/narratives/NARR_B00.md
sbatch -p normal scripts/slurm_hf_download.sh lab/config/narratives/matrix.json
squeue -u $USER                                            # empty: the batch is over
grep -lE "exit status [1-9]|Traceback" output/logs/narr-*/*/*.out   # failed pairs runs, if any
tail -1 lab/output/logs/hf_download_<job>.out              # failed: none
```

If the download fails for lack of network on the compute nodes, say so before downloading anywhere else.

**Step 2. Batch 01, the pilot E0** (1 to 2 hours): freeze its pairs on a CPU node, then queue its three chained GPU jobs:

```bash
sbatch -p normal scripts/slurm_matrix_prepare.sh lab/config/narratives/matrix_e0.json lab/output/results/narr-e0/pilot \
    lab/output/results/narr-tcr64_dce_lclsf lab/output/results/narr-bbbp_dce_lclsf
tail -1 lab/output/logs/matrix_prepare_<job>.out           # Matrix: .../narr-e0/pilot
python scripts/paper_probe_matrix.py submit --run-root lab/output/results/narr-e0/pilot
squeue -u $USER
cut -d, -f1-6 lab/output/results/narr-e0/pilot/matrix-summary.csv                  # every probe and condition, n and n_success
sacct -j $(grep -o '"job_id": "[0-9]*"' lab/output/results/narr-e0/pilot/jobs.jsonl | grep -o '[0-9]*' | paste -sd,) --format=JobName%25,State,Elapsed
```

**Step 3. Check E0 at home** (step 6 brings it): parse rates and statuses per probe, and the elapsed times. The pilot has 20 instances of one narrator; the paper has about 300 instances and five narrators. Rescale: a generator job takes about 15 times the pilot's generator time (minus the model loading, a few minutes), the primary judge about 75 times the pilot's judge time. A batch whose estimate goes over two days is split with `--only` before it is submitted.

**Step 4. Batch 02, the narratives**: freeze every pair into the paper's run, then the five generators side by side:

```bash
sbatch -p normal scripts/slurm_matrix_prepare.sh lab/config/narratives/matrix.json lab/output/results/narr-matrix/main \
    lab/output/results/narr-tcr64_dce_lclsf lab/output/results/narr-bbbp_dce_lclsf lab/output/results/narr-aids_dce_lclsf
tail -1 lab/output/logs/matrix_prepare_<job>.out           # Matrix: .../narr-matrix/main
python scripts/paper_probe_matrix.py submit --run-root lab/output/results/narr-matrix/main \
    --only generator:qwen2.5-3b,generator:qwen2.5-7b,generator:qwen2.5-14b,generator:qwen2.5-32b,generator:llama3.1-8b
```

**Step 5. Batch 03 and batch 04**, each once `squeue -u $USER` is empty:

```bash
python scripts/paper_probe_matrix.py submit --run-root lab/output/results/narr-matrix/main \
    --only judge:gpt-oss-120b,judge:gpt-oss-120b-sweep,judge:glm-4.7-flash
python scripts/paper_probe_matrix.py submit --run-root lab/output/results/narr-matrix/main \
    --only self:qwen2.5-3b,self:qwen2.5-7b,self:qwen2.5-14b,self:qwen2.5-32b,self:llama3.1-8b
```

A batch can also be queued before the previous one ends with `--after <job>` (the last job id of `jobs.jsonl`, or the ids joined with `:`). A job that failed is queued again into the same run once its `error.json` (under `probes/`) or its generated folder is removed and its line deleted from `jobs.jsonl`, which `submit` checks. `python scripts/paper_probe_matrix.py summary --run-root <run>` reads nothing but JSON and rewrites `matrix-summary.csv` at any time.

**Step 6. Back home**, after every batch, on the local machine:

```bash
rsync -av kalifano:projects/GRETEL/lab/output/results/narr-* lab/output/results/
rsync -av kalifano:projects/GRETEL/output/logs/ lab/output/server_logs/ && python scripts/collect_logs.py lab/output/server_logs
git add lab/output/results/narr-* lab/output/queue_logs && git commit -m "Narrative paper: batch <n>" && git push
```

The pairs dumps take about 300 KB per instance (18 KB in the git pack); the whole matrix, about 1.5 GB on disk and 100 to 150 MB in the pack.
