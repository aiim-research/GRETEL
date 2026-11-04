# Narrative paper: running the experiments on the server

What `docs/narratives/EXPERIMENTS.md` plans, as Slurm jobs on kalifano. Two kinds of work:

- **Pairs pass** (CPU, GRETEL): one run per dataset, fold 0, DCE + LBS, no LLM. Every instance gives three graph pairs, the arms `minimized` (LBS), `seed` (DCE) and `random_subset` (the input plus a random subset of the seed's edits of the minimized size), and freezes every generation prompt: direct, inverse, renamed (shown nodes permuted, permutation saved) and identity (the direct prompt again, sampled with seed + 1). The random subset only gets the direct prompt. Configs: `lab/config/narratives/pairs/{tcr64,bbbp,aids}/generate_minimize0.jsonc`.
- **Model matrix** (GPU, `scripts/paper_probe_matrix.py`): the pairs are frozen into a run, and each model is one Slurm job on `compute-2-3`, chained so one model uses the GPU at a time. Generator jobs write the narratives, the judge jobs run the probes over every generator's narratives, and the self jobs let each narrator judge its own Consistency inversion test (E6). Configs: `lab/config/narratives/matrix.json` (the paper) and `matrix_e0.json` (the pilot E0: the first 10 pairs of Tree-Cycles and BBBP, Qwen2.5-7B, the primary judge).

Models (decisions of 3 November 2026; API models are left for later): narrators Qwen2.5-Instruct 3B, 7B, 14B and 32B and Llama-3.1-8B-Instruct in bf16 with vLLM, temperature 0.7, top-p 0.95, 1024 output tokens, seed 0 (identity: seed 1); judge GPT-OSS-120B (local, the only judge until there is API access), sampled at temperature 1 with low reasoning effort as the pilots recommend. Exact revisions in the configs.

Probes per narrator and pair: Reconstruction under CTX on/off x MODE dict/full on the three arms; Consistency (inversion, renaming, identity) and Recourse with and without explanation at K = the instance's minimized edit count, on the seed and minimized arms. The Recourse budget sweep (K in 1, 2, 3, 5, 8, minimized arm, Tree-Cycles and BBBP) is a second judge entry, `gpt-oss-120b-sweep`.

| Batch | Jobs | Estimate | Content |
|---|---|---|---|
| B00 | 3 CPU configs + 1 download | under 3 h | pairs pass of Tree-Cycles 64, BBBP and AIDS (E1 to E3 inputs); model weights into `~/hf_cache` |
| B01 | 1 CPU + 3 GPU | 1 to 2 h | E0 pilot: Qwen2.5-7B narrates 20 instances, primary judge, self-evaluation |
| B02 | 1 CPU + 5 GPU | 3 to 8 h | E1 and the E3 narratives: the five narrators over every pair (13,500 generations) |
| B03 | 1 GPU | 6 to 18 h | E2 and E3 scoring: primary judge, about 33,000 calls |
| B04 | 6 GPU | 2 to 6 h | E4 budget sweep (10,000 calls) and E6 self-evaluation (2,000 calls) |

The estimates come from the call counts and an assumed vLLM throughput on one H100-class GPU, not from measurements: B01 measures them (step 2), and a batch whose rescaled estimate goes over two days is split with `--only` before it is submitted. Not in any batch yet: E5 (robustness judges) and the Gemini narrator need API access; E7, E8 and the analyses run offline on the local machine.

Rules: on the server only Slurm jobs compute. `launchers/slurm_submit.sh` and `python scripts/paper_probe_matrix.py submit` only list configs or read the run's JSON and call `sbatch`; `prepare`, which imports the framework, runs as a CPU job (`scripts/slurm_matrix_prepare.sh`). The narrative work lives in its own clone so the selector's clone is never switched. Each GPU job waits for 115 GB free on the GPU of `compute-2-3` for up to 6 hours (`min_free_mb`, `max_wait`), then fails with exit status 3 if it never got it.

**Step 0. Once: the narrative clone** (git, a file copy and environment checks only):

```bash
cd ~/projects/GRETEL && git fetch
git worktree add --track -b dev-narrative ../GRETEL-narrative origin/dev-narrative
cd ../GRETEL-narrative
mkdir -p data/working && cp -r ../GRETEL/data/working/AIDS data/working/   # AIDS raw data: the compute nodes need no network
conda activate GRTL && conda list vllm        # the GPU jobs need vLLM in this environment (another one: export GRETEL_ENV=<env>)
sinfo -s                                      # partitions: the GPU jobs use cuda and compute-2-3 (lab/config/narratives/matrix*.json, "slurm")
```

Llama-3.1-8B is gated: accept its license on huggingface.co with the account of the token, then `huggingface-cli login` once (or `export HF_TOKEN=...` before the download job).

**Step 1. Batch 00** (CPU, under 3 hours), the pairs and the weights in parallel:

```bash
cd ~/projects/GRETEL-narrative && launchers/server_sync.sh && conda activate GRTL
launchers/slurm_submit.sh -p normal -m 3 docs/narratives/NARR_B00.md
sbatch -p normal scripts/slurm_hf_download.sh lab/config/narratives/matrix.json
squeue -u $USER                                            # empty: the batch is over
grep -lE "exit status [1-9]|Traceback" output/logs/narr-*/*/*.out   # failed pairs runs, if any
tail -1 lab/output/logs/hf_download_<job>.out              # failed: none
```

If the download fails for lack of network on the compute nodes, say so before downloading anywhere else.

**Step 2. Batch 01, the pilot E0** (1 to 2 hours): freeze its pairs on a CPU node, then queue its three GPU jobs:

```bash
sbatch -p normal scripts/slurm_matrix_prepare.sh lab/config/narratives/matrix_e0.json lab/output/results/narr-e0/pilot \
    lab/output/results/narr-tcr64_dce_lcls lab/output/results/narr-bbbp_dce_lcls
tail -1 lab/output/logs/matrix_prepare_<job>.out           # Matrix: .../narr-e0/pilot
python scripts/paper_probe_matrix.py submit --run-root lab/output/results/narr-e0/pilot
squeue -u $USER
cut -d, -f1-6 lab/output/results/narr-e0/pilot/matrix-summary.csv                  # every probe and condition, n and n_success
sacct -j $(grep -o '"job_id": "[0-9]*"' lab/output/results/narr-e0/pilot/jobs.jsonl | grep -o '[0-9]*' | paste -sd,) --format=JobName%25,State,Elapsed
```

Bring it home (step 6) and check it locally before going on: parse rates and statuses per probe, and the elapsed times. The pilot has 20 instances of one narrator; the paper has about 300 instances and five narrators. Rescale: a generator job takes about 15 times the pilot's generator time (minus the model loading, a few minutes, paid once), the primary judge about 75 times the pilot's judge time.

**Step 3. Batch 02, the narratives** (3 to 8 hours): freeze every pair into the paper's run, then the five generator jobs:

```bash
sbatch -p normal scripts/slurm_matrix_prepare.sh lab/config/narratives/matrix.json lab/output/results/narr-matrix/main \
    lab/output/results/narr-tcr64_dce_lcls lab/output/results/narr-bbbp_dce_lcls lab/output/results/narr-aids_dce_lcls
tail -1 lab/output/logs/matrix_prepare_<job>.out           # Matrix: .../narr-matrix/main
python scripts/paper_probe_matrix.py submit --run-root lab/output/results/narr-matrix/main \
    --only generator:qwen2.5-3b,generator:qwen2.5-7b,generator:qwen2.5-14b,generator:qwen2.5-32b,generator:llama3.1-8b
```

**Step 4. Batch 03, the primary judge** (6 to 18 hours):

```bash
python scripts/paper_probe_matrix.py submit --run-root lab/output/results/narr-matrix/main --only judge:gpt-oss-120b
```

**Step 5. Batch 04, budget sweep and self-evaluation** (2 to 6 hours):

```bash
python scripts/paper_probe_matrix.py submit --run-root lab/output/results/narr-matrix/main \
    --only judge:gpt-oss-120b-sweep,self:qwen2.5-3b,self:qwen2.5-7b,self:qwen2.5-14b,self:qwen2.5-32b,self:llama3.1-8b
```

A batch can also be queued before the previous one ends: `--after <job>`, with the last job id of `jobs.jsonl`. A job that failed is queued again into the same run once its `error.json` (under `probes/`) or its generated folder is removed; `submit` refuses jobs that the run's `jobs.jsonl` already lists, so delete that line too. `python scripts/paper_probe_matrix.py summary --run-root <run>` reads nothing but JSON and rewrites `matrix-summary.csv` at any time.

**Step 6. Back home**, after every batch, on the local machine:

```bash
rsync -av kalifano:projects/GRETEL-narrative/lab/output/results/narr-* lab/output/results/
rsync -av kalifano:projects/GRETEL-narrative/output/logs/ lab/output/server_logs/ && python scripts/collect_logs.py lab/output/server_logs
git add lab/output/results/narr-* lab/output/queue_logs && git commit -m "Narrative paper: batch <n>" && git push
```

The pairs dumps take about 300 KB per instance (18 KB in the git pack); the whole matrix, about 1.5 GB on disk and 100 to 150 MB in the pack.
