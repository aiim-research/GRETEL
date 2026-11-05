# Narrative paper, Gianrico's account

Your half of the model matrix: the narrators Qwen2.5-3B, 7B and 14B, judged by GPT-OSS-120B (primary, and the Recourse budget sweep) and GLM-4.7-Flash, plus the judge validation V0 after the batch. Rodrigo runs Qwen2.5-32B and Llama-3.1-8B with the same judges from his account, on the same frozen pairs. Overview and models: `RUNS.md`.

Rule of the server: only Slurm jobs compute. The commands below only use git, `sbatch`/`srun`, or read JSON.

**Step 0. Once: the code, the GPUs, the environment, Hugging Face.**

```bash
cd ~/projects && git clone -b dev https://github.com/aiim-research/GRETEL.git GRETEL-dev && cd GRETEL-dev
git remote add fork https://github.com/GianFor/GRETEL.git        # where your results go (step 6)
srun -p cuda --gres=gpu:fast:1 -c 1 --mem=1G nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

If you already have a clone, `git fetch origin && git switch -c dev origin/dev` in it works too. The driver must be 580 or newer for vLLM 0.33.0; if not, tell Rodrigo before going on.

The GPU environment `GRTL-llm`, apart from your `GRTL` (Python 3.12, vLLM 0.33.0, the same version as Rodrigo's, so the two halves are comparable). A CPU job with network, about half an hour and 15 GB:

```bash
sbatch -p normal --export=ALL,VLLM_VERSION=0.33.0 scripts/slurm_setup_llm_env.sh
tail -3 lab/output/logs/llm_env_<job>.out        # import check failed: none / environment GRTL-llm ready
```

Your models are all public, so no license is needed. A Read token avoids download rate limits (huggingface.co, Settings, Access Tokens):

```bash
conda activate GRTL-llm && hf auth login         # answer n to "add as git credential"
df -h ~ ; quota -s                               # about 190 GB of weights go to ~/hf_cache
```

**Step 1. CPU: the weights and the AIDS raw data**, while Rodrigo makes the pairs:

```bash
sbatch -p normal scripts/slurm_hf_download.sh lab/config/narratives/matrix_g.json
sbatch -p normal scripts/slurm_hf_download.sh lab/config/narratives/matrix_v0.json     # GPT-OSS-20B, for V0
sbatch -p normal --wrap "bash -lc 'conda activate GRTL && python tools/warm_dataset.py lab/config/narratives/pairs/aids/generate_minimize0.jsonc'"
squeue -u $USER
tail -1 lab/output/logs/hf_download_*.out        # failed: none
```

The last job downloads the AIDS raw data into `data/working` (the Recourse probe rebuilds the molecules to ask the oracle); the BBBP and Tree-Cycles data are in the repository.

**Step 2. When Rodrigo says the pairs are in `dev`:**

```bash
git pull
ls lab/output/results/                           # narr-tcr64_dce_lclsf  narr-bbbp_dce_lclsf  narr-aids_dce_lclsf
```

**Step 3. Freeze your three runs** (CPU jobs, minutes each):

```bash
for d in tcr64 bbbp aids; do
  sbatch -p normal scripts/slurm_matrix_prepare.sh lab/config/narratives/matrix_g.json \
      lab/output/results/narr-matrix/$d-g lab/output/results/narr-${d}_dce_lclsf
done
squeue -u $USER
tail -qn1 lab/output/logs/matrix_prepare_*.out   # Matrix: .../narr-matrix/<dataset>-g, three times
```

**Step 4. Launch everything** and come back in about two days:

```bash
conda activate GRTL
python scripts/paper_probe_matrix.py fleet --max 2 \
    --run-root lab/output/results/narr-matrix/tcr64-g \
    --run-root lab/output/results/narr-matrix/bbbp-g \
    --run-root lab/output/results/narr-matrix/aids-g
```

This is the same mechanism as the fixed-size Slurm queue: each run is one array of narrators and one array of judges and self-evaluation, at most `--max` tasks of each running at a time, the next ones starting as the current ones end. Tree-Cycles is narrated first, then BBBP, then AIDS, and the judges of a dataset start when its narrators end, while the next dataset is narrated. `--max 2`: there are five `fast` GPUs and Rodrigo's account uses them too.

**Step 5. While it runs:**

```bash
squeue -u $USER
python scripts/paper_probe_matrix.py summary --run-root lab/output/results/narr-matrix/tcr64-g    # reads JSON only
cut -d, -f1-6 lab/output/results/narr-matrix/tcr64-g/matrix-summary.csv
grep -l "Traceback" lab/output/logs/slurm_*.out
```

Tree-Cycles ends first: check its summary (parse rates per probe) before BBBP and AIDS use their GPU time, and send it to Rodrigo. A failed task is queued again with `python scripts/paper_probe_matrix.py submit --run-root <run> --only <role>:<id>` once its line is deleted from `<run>/jobs.jsonl` and its `generated/<id>` or `probes/*/<id>/error.json` is removed.

**Step 6. The results to `dev`**, when everything is over:

```bash
git switch -c narr-results-gianrico
git add lab/output/results/narr-matrix && git commit -m "Narrative paper: Gianrico's half of the matrix"
git push fork narr-results-gianrico
```

Then open a pull request from `GianFor:narr-results-gianrico` into `aiim-research:dev`.

**Step 7. After the batch: the judge validation V0.** Synthetic narrators write narratives with known content (all the edits, half of them, invented ones, none; a true, false, static, contradicting or absent mechanism) on 20 pairs of Tree-Cycles and 20 of BBBP, and GPT-OSS-120B, GLM-4.7-Flash and GPT-OSS-20B judge them. The synthetic narrators run on CPU nodes, the judges on GPUs.

```bash
git switch dev && git pull
sbatch -p normal scripts/slurm_matrix_prepare.sh lab/config/narratives/matrix_v0.json lab/output/results/narr-v0/main \
    lab/output/results/narr-tcr64_dce_lclsf lab/output/results/narr-bbbp_dce_lclsf
python scripts/paper_probe_matrix.py fleet --max 3 --run-root lab/output/results/narr-v0/main
squeue -u $USER                                  # when it is empty:
sbatch -p normal --wrap "bash -lc 'conda activate GRTL && python scripts/v0_judge_validation.py --run-root lab/output/results/narr-v0/main'"
cat lab/output/results/narr-v0/main/v0-report.md
```

The report gives, per judge, the Reconstruction fidelity (the extraction is exactly what the text states), the Consistency accuracy against the expected verdicts, and the Recourse success with the true mechanism, a false one and none. Commit it with the run like step 6.
