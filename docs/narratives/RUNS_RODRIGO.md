# Narrative paper, Rodrigo's account

The pairs pass for both accounts, then the narrators Qwen2.5-32B and Llama-3.1-8B with every judge. Overview and models: `RUNS.md`. Gianrico runs the other narrators in parallel (`RUNS_GIANRICO.md`) once the pairs are in `dev` (end of step 2).

**Step 0. Once: GPUs, environment, Hugging Face, disk.** The clone is on `dev`.

```bash
cd ~/projects/GRETEL && launchers/server_sync.sh
srun -p cuda --gres=gpu:fast:1 -c 1 --mem=1G nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

vLLM 0.33.0 needs a driver 580 or newer; with an older driver or cards under about 100 GB, stop here and adjust the configs first.

**Step 0b. Create the GPU environment `GRTL-llm`.** The narrators and judges run with vLLM, which `GRTL` does not have and cannot get without changing its versions, so the GPU jobs use a second conda environment. `scripts/slurm_setup_llm_env.sh` creates it inside a CPU job (it needs network on the node; about half an hour, 15 GB): `conda create -n GRTL-llm python=3.12` from conda-forge, then `pip install vllm==0.33.0` (which brings its own torch for CUDA 13) plus torch_geometric, RDKit and the rest of what GRETEL needs, then an import of every module the GPU jobs load, and the exact versions in `lab/output/logs/llm_env_freeze_<job>.txt`. Gianrico builds his with the same `VLLM_VERSION`. The matrix configs name it (`"env": "GRTL-llm"`), so the GPU jobs activate it by themselves.

```bash
sbatch -p normal --export=ALL,VLLM_VERSION=0.33.0 scripts/slurm_setup_llm_env.sh
tail -3 lab/output/logs/llm_env_<job>.out        # import check failed: none / environment GRTL-llm ready
```

Llama-3.1-8B-Instruct is gated: accept its license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct (approval by e-mail), create a Read token (Settings, Access Tokens) and store it once:

```bash
conda activate GRTL-llm && hf auth login && hf auth whoami      # answer n to "add as git credential"
df -h ~ ; quota -s                                               # about 220 GB of weights go to ~/hf_cache
```

**What the pairs are.** For each fold-0 instance of a dataset (an input graph), the counterfactuals the narrators will explain, in three arms: the DCE seed, its BLS-F minimization, and a random subset of the seed's edits of the minimized size (the size control). One GRETEL run per dataset (DCE + BLS-F, no LLM, `lab/config/narratives/pairs/<dataset>/`) writes one JSON per instance under `lab/output/results/narr-<dataset>_dce_lclsf/.../probe_inputs/`, with both graphs of every arm, the oracle's classes, the true typed edit set (the ground truth of the probes), the renaming permutation, the Recourse budget K, and the exact prompts every narrator will receive (direct, inverse, renamed, identity). `prepare` copies them into each run of the matrix with their hashes, so both accounts narrate and judge exactly the same inputs.

**Step 1. CPU: the pairs and the weights** (a few hours):

```bash
conda activate GRTL
launchers/slurm_submit.sh -p normal -m 3 docs/narratives/NARR_B00.md
sbatch -p normal scripts/slurm_hf_download.sh lab/config/narratives/matrix_r.json
squeue -u $USER
grep -lE "exit status [1-9]|Traceback" output/logs/narr-*/*/*.out      # failed pairs runs, if any
tail -1 lab/output/logs/hf_download_<job>.out                         # failed: none
for d in tcr64 bbbp aids; do echo $d $(find lab/output/results/narr-${d}_dce_lclsf -name 'cf_*.json' | wc -l); done   # about 100 each
```

The protocol v2 sondeo (`docs/v2/RUNS.md`) can run in the same CPU window.

**Step 2. At home: the pairs into `dev`**, so Gianrico starts from the same ones:

```bash
rsync -av kalifano:projects/GRETEL/lab/output/results/narr-{tcr64,bbbp,aids}_dce_lclsf lab/output/results/
git add lab/output/results/narr-*_dce_lclsf && git commit -m "Narrative paper: the pairs of Tree-Cycles 64, BBBP and AIDS" && git push
```

> **MARK: tell Gianrico now that the pairs are in `dev` and he can start his step 2.**

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

**Step 4. The batches, one per dataset**, each launched when the previous one is over (`squeue -u $USER` empty):

```bash
python scripts/paper_probe_matrix.py fleet --max 2 --run-root lab/output/results/narr-matrix/tcr64-r    # B1, Tree-Cycles
python scripts/paper_probe_matrix.py fleet --max 2 --run-root lab/output/results/narr-matrix/bbbp-r     # B2, BBBP
python scripts/paper_probe_matrix.py fleet --max 2 --run-root lab/output/results/narr-matrix/aids-r     # B3, AIDS
```

`fleet` is to the model matrix what `slurm_submit.sh` is to the GRETEL configs: per run, one array of narrators and one of judges and self-evaluation that starts when the narrators' array has ended, at most `--max` tasks of each running at a time. B1 gives the real times; if it is much shorter than two days, B2 and B3 go together in one command (`--run-root .../bbbp-r --run-root .../aids-r`, chained in that order). `--max 2`: five `fast` GPUs shared with Gianrico's account.

**Step 5. While a batch runs:**

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
