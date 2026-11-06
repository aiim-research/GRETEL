# Narrative paper, the model matrix (Gianrico's account)

Every GPU run of the paper, from your account (512 GB of quota): five narrators and their judges over the frozen pairs that Rodrigo computed and pushed to `dev`. Overview, models and probes: `RUNS.md`.

Rule of the server: only Slurm jobs compute. The commands below only use git, `sbatch`/`srun`, or read JSON.

**Step 0. Once: the code and the GPU environment.**

```bash
cd ~/projects && git clone -b dev https://github.com/aiim-research/GRETEL.git GRETEL-dev && cd GRETEL-dev
git remote add fork https://github.com/GianFor/GRETEL.git        # where your results go (step 5)
ls lab/output/results/                                           # narr-tcr64_dce_lclsf  narr-bbbp_dce_lclsf  narr-aids_dce_lclsf
```

If you already have a clone, `git fetch origin && git switch -c dev origin/dev` in it works too.

Create the GPU environment `GRTL-llm`, apart from your `GRTL`. `scripts/slurm_setup_llm_env.sh` creates it inside a CPU job (it needs network on the node; about half an hour and 10 GB): `conda create -n GRTL-llm python=3.12` from conda-forge, then `pip install vllm==0.33.0` (built for CUDA 13; the H200 nodes have driver 590) plus torch_geometric, RDKit and the rest of what GRETEL needs, then an import of every module the GPU jobs load. The matrix configs name it, so the GPU jobs activate it by themselves. Rodrigo built the same one on 5 November without problems.

```bash
sbatch -p normal --export=ALL,VLLM_VERSION=0.33.0 scripts/slurm_setup_llm_env.sh
tail -3 lab/output/logs/llm_env_<job>.out        # import check failed: none / environment GRTL-llm ready
conda activate GRTL-llm && pip cache purge && conda clean -a -y   # about 7 GB of installation caches
```

**Step 1. The weights and the AIDS raw data.** The narrators are the ones of your pilots, at the same revisions: Qwen3.5 2B, 4B, 9B and 27B and Gemma-4-31B-it, already in `~/hf_cache` like GLM-4.7-Flash and GPT-OSS-20B. The only new model is GPT-OSS-120B, the primary judge (65 GB). With the environment (10 GB) that needs about 80 GB free, so check first:

```bash
df -h ~
du -sh ~/hf_cache/hub/models--* | sort -h        # what is in the cache
```

If there is not enough room, Qwen3.8-27B (56 GB) and Muse-Glimmer-30B (60 GB) are not used by these runs and can go (`rm -rf ~/hf_cache/hub/models--Qwen--Qwen3.8-27B`, and the same for `models--meta-models--Muse-Glimmer-30B`); they can be downloaded again later. Then the download, which skips what is already there, and the AIDS raw data (the Recourse probe rebuilds the molecules to ask the oracle; BBBP and Tree-Cycles come with the repository):

```bash
sbatch -p normal scripts/slurm_hf_download.sh lab/config/narratives/matrix.json
sbatch -p normal --wrap "bash -lc 'conda activate GRTL && python tools/warm_dataset.py lab/config/narratives/pairs/aids/generate_minimize0.jsonc'"
tail -1 lab/output/logs/hf_download_<job>.out    # failed: none
```

**Step 2. Freeze the three runs** (CPU jobs, minutes each): each copies the frozen pairs of one dataset into its run, with their hashes.

```bash
for d in tcr64 bbbp aids; do
  sbatch -p normal scripts/slurm_matrix_prepare.sh lab/config/narratives/matrix.json \
      lab/output/results/narr-matrix/$d lab/output/results/narr-${d}_dce_lclsf
done
squeue -u $USER
tail -qn1 lab/output/logs/matrix_prepare_*.out   # Matrix: .../narr-matrix/<dataset>, three times
```

**Step 3. The batches, one per dataset**, each launched when the previous one is over (`squeue -u $USER` empty):

```bash
conda activate GRTL
python scripts/paper_probe_matrix.py fleet --max 3 --run-root lab/output/results/narr-matrix/tcr64    # B1, Tree-Cycles
python scripts/paper_probe_matrix.py fleet --max 3 --run-root lab/output/results/narr-matrix/bbbp     # B2, BBBP
python scripts/paper_probe_matrix.py fleet --max 3 --run-root lab/output/results/narr-matrix/aids     # B3, AIDS
```

`fleet` is the fixed-size Slurm queue for the model matrix: for a run it submits one array with the five narrators and one with the judges and the self-evaluation, which starts when the narrators' array has ended; at most `--max` tasks of an array run at a time, the next ones starting as the current ones end. B1 also gives the real times: if it takes much less than two days, B2 and B3 can go together in one command (`--run-root .../bbbp --run-root .../aids`, chained in that order).

**Step 4. While a batch runs:**

```bash
squeue -u $USER
python scripts/paper_probe_matrix.py summary --run-root lab/output/results/narr-matrix/tcr64    # reads JSON only
cut -d, -f1-6 lab/output/results/narr-matrix/tcr64/matrix-summary.csv
grep -l "Traceback" lab/output/logs/slurm_*.out
```

After B1, send Rodrigo the Tree-Cycles summary (parse rates per probe) and the elapsed times (`sacct -u $USER -S <date> --format=JobName%25,State,Elapsed`) before launching B2. A failed task is queued again with `python scripts/paper_probe_matrix.py submit --run-root <run> --only <role>:<id>` once its line is deleted from `<run>/jobs.jsonl` and its `generated/<id>` or `probes/*/<id>/error.json` is removed.

**Step 5. The results to `dev`**, after each batch:

```bash
git switch -c narr-results                       # the first time; then git switch narr-results
git add lab/output/results/narr-matrix && git commit -m "Narrative paper: model matrix, <dataset>"
git push fork narr-results
```

Then open (or update) a pull request from `GianFor:narr-results` into `aiim-research:dev`.
