# Narrative paper: running the experiments on the server

What `docs/narratives/EXPERIMENTS.md` plans, as Slurm jobs on kalifano. The step-by-step tutorials:

- `RUNS_GIANRICO.md`: every GPU run (the model matrix on shared GPUs), from Gianrico's account, whose quota holds the weights.
- `RUNS_RODRIGO.md`: the pairs pass (done) and how to make it again.

## The pipeline

1. **Pairs pass** (CPU, GRETEL, protocol v2: no padding, isolated nodes deleted, `docs/v2/README.md`), done once in Rodrigo's account and committed to `dev`. One run per dataset, fold 0, DCE + BLS-F, no LLM. Every instance gives three graph pairs, the arms `minimized` (BLS-F: bond edits and atom substitutions), `seed` (DCE's nearest graph of the other class, written as edits on the input) and `random_subset` (the input plus a random subset of the seed's edits of the minimized size), and freezes every generation prompt: direct, inverse, renamed and identity (the direct prompt again, sampled with seed + 1). Results: `lab/output/results/narr-<dataset>_dce_lclsf` (Tree-Cycles 64: 100 instances, BBBP: 102, AIDS: 100).
2. **Model matrix** (GPU, `scripts/paper_probe_matrix.py`, plan C of 16 November: `EXPERIMENTS.md`, section 10), one run per dataset with every model: `lab/config/narratives/matrix_local.json`. Tree-Cycles is Gianrico's run `narr-matrix/tcr64-20b-h200-longwait` (completed, not repeated), BBBP and AIDS are `lab/output/results/narr-local/<dataset>`. The GPUs are shared, so each model waits only for its own memory and the models are queued by memory class (`fleet --tier s`, `m`, `l`). A task is one model over the three datasets, loaded once. The narrators write the narratives, the judges run the probes over them, and each narrator judges its own Consistency inversion test (E6). Order: a load test of every model (`fit`), class `s` on Tree-Cycles alone (the sanity check and the real times), then class `s` on the three datasets and class `m`, then class `l`. Every command can be launched again and only does what is missing.
3. **Judge validation V0**, after the batches, prepared by Rodrigo and Gianrico together (annotated narratives, and synthetic ones with known content: `src/LLMexplaneability/synthetic.py`, `lab/config/narratives/matrix_v0.json`, `scripts/v0_judge_validation.py`). Its judges are the matrix's, with the same budgets. If the primary judge fails a probe, only the judges are run again, never the narrators.

## Models and probes

Decided on 6 November: the narrators are the models of Gianrico's pilots, newer than the draft's list (Qwen2.5 and Llama-3.1) and already downloaded. Decided on 16 November: everything local for now, each model with its own GPU budget. Decided on 17 November: Gianrico's Tree-Cycles run is reused as it is, so every model runs as there, in bf16 (GPT-OSS as stored), with budgets measured in his logs. Gemma-4-31B-it needs 93 GiB, most of it KV cache, so it only runs on an H200.

| Role | Models | Class, GiB | As loaded |
|---|---|---|---|
| Narrators, scale ladder | Qwen3.5 2B, 4B, 9B | `s`: 12, 18, 28 | bf16 |
| Narrators, scale ladder | Qwen3.5-27B | `m`: 60 | bf16 |
| Narrator of another family | Gemma-4-31B-it | `l`: 93 (H200 only) | bf16 |
| Judge of the three probes and of the Recourse budget sweep, now | GPT-OSS-20B | `s`: 22 | MXFP4, as stored |
| Robustness judge | GLM-4.7-Flash | `m`: 64 | bf16 |
| Reference judge of the three probes and of the sweep, when a GPU has 71 GiB free | GPT-OSS-120B | `l`: 70 | MXFP4, as stored |
| Later, with API access | DeepSeek-V3.2 and Gemini 3 Flash (judges), Gemini 2.5 Flash (narrator) | | |

Narrators: vLLM, thinking off and no image or video input, temperature 0.7, top-p 0.95, 4096 output tokens (1024 cut half of the valid answers in the Tree-Cycles run of 17 November), seed 0 (identity: seed 1), context 40 960 tokens (the longest narrator prompts reach about 35 000). No judge reads narratives of its own family. Probes per narrator and pair (GPT-OSS-20B and GPT-OSS-120B): Reconstruction under CTX on/off x MODE dict/full on the three arms. Consistency (inversion, renaming, identity) and Recourse with and without explanation at K = the instance's minimized edit count, on the seed and minimized arms. The sweep, K in 1, 2, 3, 5, 8 on the minimized arm of Tree-Cycles and BBBP. GLM-4.7-Flash: Reconstruction CTX-off/MODE-dict, Consistency and Recourse. If GPT-OSS-120B runs it is the primary judge and GPT-OSS-20B a second size of its family. If it never finds the memory, GPT-OSS-20B is the primary judge.

## Rules

On the server only Slurm jobs compute. `launchers/slurm_submit.sh` and `paper_probe_matrix.py fleet`, `fit`, `submit` and `extend` only read JSON and call `sbatch`, and `summary` and `pending` only read JSON. `prepare`, which imports the framework, runs as a CPU job (`scripts/slurm_matrix_prepare.sh`). Each GPU task waits until its GPU has its model's budget plus 1 GiB free, for up to 8 days, and fails with exit status 3 if it never does. One task of ours at a time checks a GPU's memory and loads on it (a lock per GPU), so two tasks never start on the same free memory.
