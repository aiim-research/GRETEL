# Narrative paper, Gianrico's account

Your half of the model matrix: the narrators Qwen2.5-3B, 7B and 14B, judged by GPT-OSS-120B (primary, and the Recourse budget sweep) and GLM-4.7-Flash. Rodrigo runs Qwen2.5-32B and Llama-3.1-8B with the same judges from his account, on the same frozen pairs. Overview and models: `RUNS.md`.

Rule of the server: only Slurm jobs compute. The commands below only use git, `sbatch`/`srun`, or read JSON.

**Step 0. Once: the code, the GPUs, the environment, Hugging Face.**

```bash
cd ~/projects && git clone -b dev https://github.com/aiim-research/GRETEL.git GRETEL-dev && cd GRETEL-dev
git remote add fork https://github.com/GianFor/GRETEL.git        # where your results go (step 6)
srun -p cuda --gres=gpu:fast:1 -c 1 --mem=1G nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

If you already have a clone, `git fetch origin && git switch -c dev origin/dev` in it works too. The driver must be 580 or newer for vLLM 0.33.0; if not, tell Rodrigo before going on.

**Create the GPU environment `GRTL-llm`**, apart from your `GRTL`: the narrators and judges need vLLM, which `GRTL` cannot get without changing its versions. `scripts/slurm_setup_llm_env.sh` creates it inside a CPU job (it needs network on the node; about half an hour, 15 GB): `conda create -n GRTL-llm python=3.12` from conda-forge, then `pip install vllm==0.33.0` (the same version as Rodrigo's, so the two halves are comparable) plus torch_geometric, RDKit and the rest of what GRETEL needs, then an import of every module the GPU jobs load. The matrix configs name it, so the GPU jobs activate it by themselves.

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

**Step 4. The batches, one per dataset**, each launched when the previous one is over (`squeue -u $USER` empty):

```bash
conda activate GRTL
python scripts/paper_probe_matrix.py fleet --max 2 --run-root lab/output/results/narr-matrix/tcr64-g    # B1, Tree-Cycles
python scripts/paper_probe_matrix.py fleet --max 2 --run-root lab/output/results/narr-matrix/bbbp-g     # B2, BBBP
python scripts/paper_probe_matrix.py fleet --max 2 --run-root lab/output/results/narr-matrix/aids-g     # B3, AIDS
```

`fleet` is the fixed-size Slurm queue for the model matrix: for a run it submits one array with the narrators and one with the judges and the self-evaluation, which starts when the narrators' array has ended; at most `--max` tasks of an array run at a time, the next ones starting as the current ones end. `--max 2`: there are five `fast` GPUs and Rodrigo's account uses them too. B1 also gives the real times: if it takes much less than two days, B2 and B3 can go together in one command (`fleet` chains the runs in the order given: `--run-root .../bbbp-g --run-root .../aids-g`).

**Step 5. While a batch runs:**

```bash
squeue -u $USER
python scripts/paper_probe_matrix.py summary --run-root lab/output/results/narr-matrix/tcr64-g    # reads JSON only
cut -d, -f1-6 lab/output/results/narr-matrix/tcr64-g/matrix-summary.csv
grep -l "Traceback" lab/output/logs/slurm_*.out
```

After B1, send Rodrigo the Tree-Cycles summary (parse rates per probe) and the elapsed times (`sacct -u $USER -S today --format=JobName%25,State,Elapsed`) before launching B2. A failed task is queued again with `python scripts/paper_probe_matrix.py submit --run-root <run> --only <role>:<id>` once its line is deleted from `<run>/jobs.jsonl` and its `generated/<id>` or `probes/*/<id>/error.json` is removed.

**Step 6. The results to `dev`**, after each batch or at the end:

```bash
git switch -c narr-results-gianrico            # the first time; then git switch narr-results-gianrico
git add lab/output/results/narr-matrix && git commit -m "Narrative paper: Gianrico's half of the matrix, <dataset>"
git push fork narr-results-gianrico
```

Then open (or update) a pull request from `GianFor:narr-results-gianrico` into `aiim-research:dev`.
