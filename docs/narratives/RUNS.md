# Narrative paper: running the experiments on the server

What `docs/narratives/EXPERIMENTS.md` plans, as Slurm jobs on kalifano. The step-by-step tutorials:

- `RUNS_GIANRICO.md`: every GPU run (the model matrix), from Gianrico's account, whose quota holds the weights.
- `RUNS_RODRIGO.md`: the pairs pass (done) and how to make it again.

## The pipeline

1. **Pairs pass** (CPU, GRETEL, protocol v2: no padding, isolated nodes deleted, `docs/v2/README.md`), done once in Rodrigo's account and committed to `dev`. One run per dataset, fold 0, DCE + BLS-F, no LLM. Every instance gives three graph pairs, the arms `minimized` (BLS-F: bond edits and atom substitutions), `seed` (DCE's nearest graph of the other class, written as edits on the input) and `random_subset` (the input plus a random subset of the seed's edits of the minimized size), and freezes every generation prompt: direct, inverse, renamed and identity (the direct prompt again, sampled with seed + 1). Results: `lab/output/results/narr-<dataset>_dce_lclsf` (Tree-Cycles 64: 100 instances, BBBP: 102, AIDS: 100).
2. **Model matrix** (GPU, `scripts/paper_probe_matrix.py`), one run per dataset: the pairs are frozen into the run, the narrators write the narratives, the judges run the probes over them, and each narrator judges its own Consistency inversion test (E6). One batch per dataset, each launched when the previous one is over: Tree-Cycles (B1, which also gives the real times), then BBBP (B2), AIDS last (B3). A batch is one `fleet` command: an array of narrators and an array of judges and self-evaluation that starts when the narrators end, with a fixed number of tasks at a time. Runs: `lab/output/results/narr-matrix/<dataset>`.
3. **Judge validation V0**, after the batches, prepared by Rodrigo and Gianrico together (annotated narratives, and synthetic ones with known content: `src/LLMexplaneability/synthetic.py`, `lab/config/narratives/matrix_v0.json`, `scripts/v0_judge_validation.py`). If the primary judge fails a probe, only the judges are run again (`rerun-judges`), never the narrators.

## Models and probes

Decided on 6 November: the narrators are the models of Gianrico's pilots, newer than the draft's list (Qwen2.5 and Llama-3.1) and already downloaded.

| Role | Models |
|---|---|
| Narrators, scale ladder | Qwen3.5 2B, 4B, 9B, 27B |
| Narrator of another family | Gemma-4-31B-it |
| Primary judge (three probes) and Recourse budget sweep | GPT-OSS-120B |
| Robustness judge | GLM-4.7-Flash |
| V0 only | GPT-OSS-20B |
| Later, with API access | DeepSeek-V3.2 and Gemini 3 Flash (judges), Gemini 2.5 Flash (narrator) |

Narrators: bf16 with vLLM, thinking off and no image or video input, temperature 0.7, top-p 0.95, 1024 output tokens, seed 0 (identity: seed 1), context 40 960 tokens (the longest narrator prompts reach about 35 000). No judge reads narratives of its own family. Probes per narrator and pair (primary judge): Reconstruction under CTX on/off x MODE dict/full on the three arms; Consistency (inversion, renaming, identity) and Recourse with and without explanation at K = the instance's minimized edit count, on the seed and minimized arms; the sweep, K in 1, 2, 3, 5, 8 on the minimized arm of Tree-Cycles and BBBP. GLM-4.7-Flash: Reconstruction CTX-off/MODE-dict, Consistency and Recourse.

## Rules

On the server only Slurm jobs compute. `launchers/slurm_submit.sh`, `paper_probe_matrix.py fleet` and `submit` only read JSON and call `sbatch`, and `summary` only reads JSON; `prepare`, which imports the framework, runs as a CPU job (`scripts/slurm_matrix_prepare.sh`). Each GPU task waits until its GPU has 70 GB free, for up to 6 hours, and fails with exit status 3 if it never does.
