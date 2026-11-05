# Narrative paper, Rodrigo's account

The pairs pass for both accounts, then the narrators Qwen2.5-32B and Llama-3.1-8B with every judge. Overview and models: `RUNS.md`. Gianrico runs the other narrators in parallel (`RUNS_GIANRICO.md`) once the pairs are in `dev` (step 3).

**Step 0. Once: GPUs, environment, Hugging Face, disk.** The clone is on `dev`.

```bash
cd ~/projects/GRETEL && launchers/server_sync.sh
srun -p cuda --gres=gpu:fast:1 -c 1 --mem=1G nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

vLLM 0.33.0 needs a driver 580 or newer; with an older driver or cards under about 100 GB, stop here and adjust the configs first. Then the GPU environment `GRTL-llm`, the same version Gianrico builds (a CPU job with network, about half an hour, 15 GB):

```bash
sbatch -p normal --export=ALL,VLLM_VERSION=0.33.0 scripts/slurm_setup_llm_env.sh
tail -3 lab/output/logs/llm_env_<job>.out        # import check failed: none / environment GRTL-llm ready
```

Llama-3.1-8B-Instruct is gated: accept its license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct (approval by e-mail), create a Read token (Settings, Access Tokens) and store it once:

```bash
conda activate GRTL-llm && hf auth login && hf auth whoami      # answer n to "add as git credential"
df -h ~ ; quota -s                                               # about 220 GB of weights go to ~/hf_cache
```

**Step 1. CPU: the pairs and the weights** (a few hours):

```bash
conda activate GRTL
launchers/slurm_submit.sh -p normal -m 3 docs/narratives/NARR_B00.md
sbatch -p normal scripts/slurm_hf_download.sh lab/config/narratives/matrix_r.json
squeue -u $USER
grep -lE "exit status [1-9]|Traceback" output/logs/narr-*/*/*.out      # failed pairs runs, if any
tail -1 lab/output/logs/hf_download_<job>.out                         # failed: none
```

The protocol v2 sondeo (`docs/v2/RUNS.md`) can run in the same CPU window.

**Step 2. At home: the pairs into `dev`**, so Gianrico starts from the same ones:

```bash
rsync -av kalifano:projects/GRETEL/lab/output/results/narr-{tcr64,bbbp,aids}_dce_lclsf lab/output/results/
git add lab/output/results/narr-*_dce_lclsf && git commit -m "Narrative paper: the pairs of Tree-Cycles 64, BBBP and AIDS" && git push
```

Then tell Gianrico the pairs are in `dev`.

**Step 3. Freeze the three runs** (CPU jobs, minutes each):

```bash
cd ~/projects/GRETEL && launchers/server_sync.sh
for d in tcr64 bbbp aids; do
  sbatch -p normal scripts/slurm_matrix_prepare.sh lab/config/narratives/matrix_r.json \
      lab/output/results/narr-matrix/$d-r lab/output/results/narr-${d}_dce_lclsf
done
squeue -u $USER
tail -qn1 lab/output/logs/matrix_prepare_*.out                          # Matrix: .../narr-matrix/<dataset>-r, three times
```

**Step 4. Launch everything** and come back in about two days:

```bash
python scripts/paper_probe_matrix.py fleet --max 2 \
    --run-root lab/output/results/narr-matrix/tcr64-r \
    --run-root lab/output/results/narr-matrix/bbbp-r \
    --run-root lab/output/results/narr-matrix/aids-r
```

Tree-Cycles is narrated first, then BBBP, then AIDS; the judges of a dataset start when its narrators end. `--max 2` keeps at most two tasks of each array running (there are five `fast` GPUs and Gianrico's account uses them too).

**Step 5. While it runs:**

```bash
squeue -u $USER
python scripts/paper_probe_matrix.py summary --run-root lab/output/results/narr-matrix/tcr64-r    # reads JSON only
cut -d, -f1-6 lab/output/results/narr-matrix/tcr64-r/matrix-summary.csv
grep -l "Traceback" lab/output/logs/slurm_*.out
```

Tree-Cycles ends first: look at its summary early (parse rates per probe), before BBBP and AIDS have used their GPU time. A task that failed is queued again with `submit --run-root <run> --only <role>:<id>` once its line is deleted from `<run>/jobs.jsonl` and its `generated/<id>` or `probes/*/<id>/error.json` is removed.

**Step 6. At home, when it is over:**

```bash
rsync -av kalifano:projects/GRETEL/lab/output/results/narr-matrix lab/output/results/
git add lab/output/results/narr-matrix && git commit -m "Narrative paper: Rodrigo's half of the matrix" && git push
```

Gianrico's half arrives as a pull request into `dev`.
