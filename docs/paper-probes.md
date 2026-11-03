# Independent graph probes

This extension implements Reconstruction, Reversal and Recourse as a second
pass over explanations saved by a normal GRETEL run, and as GRETEL stages for
configs that want the judge in the same pipeline. TreeCycles is the first
integration dataset. The probe utilities contain no TreeCycles oracle
rules, cycle detectors or dataset-specific scoring.

The starting points are the branch's thesis-based Reconstruction,
Alejandra's `llm_explanation_contrastive_explanation` and `FlipRateEvaluator`,
and Rodrigo's proposed inverse-edit comparison and no-explanation control.
Upstream stages, evaluator, factories, explainer and oracle implementations
are used without modification. `run_paper_probes.py` reads the complete
first-pass records.

## Two passes

1. The normal Runtime invokes the configured explainer/minimizer.
   `ProbeNarratives` computes the measured delta, predicts both classes,
   and asks the configured generator for separate direct and inverse answers.
   `ProbeDump` saves every counterfactual, both answers and complete graphs.
2. Each probe can run alone against those records with an independently
   configured judge. Reconstruction/Reversal do not load an oracle.
   Recourse restores the graphs and obtains dataset/oracle through GRETEL's
   existing factories, checking both saved predictions before evaluation.

The generator is a component of `ProbeNarratives`; the judge is a component
of the probe stage or offline run. Neither uses `context.llm`. Configured
model IDs must differ. Supplying generator/judge family identifiers also
rejects a declared equal family; an omitted family cannot be inferred from
an arbitrary model name.

`ProbeNarratives` loads its generator at the first `process` call. Upstream
`MainPipeline.init` constructs its stages twice; the discarded stage objects
therefore do not load duplicate GPU models. Running generator and several
judges together in one process still needs room for each active model;
the two-pass runner loads a single judge at a time.

The paper experiments use the two passes. The stages `Reconstruction`,
`Reversal` and `Recourse` (in `src/evaluation/future/stages/`) take the
options of a probe entry of the offline config, with one `context` and `mode`
instead of the `conditions` list, plus `judge` and optionally `judge_family`.
They must follow `ProbeNarratives` in the pipeline. They call the same code as
`run_paper_probes.py`; a test checks that both give identical records.

## Shared answer and edit contracts

The generator must actually produce both fields:

```json
{
  "edits": {
    "nodes_added": [],
    "nodes_removed": [],
    "edges_added": [[1, 3]],
    "edges_removed": [],
    "features_changed": [{"node": 3, "feature": "charge", "from": 0, "to": 1}]
  },
  "Natural_Language_Explanation": "An explanation of the class change."
}
```

Three quantities remain separate: the deterministic delta from the graph
pair, the generator's declared edits, and edits independently extracted by
the judge. The true delta is never appended to the generator's answer for
MODE-full. The generator's actual answer, model configuration and prompts
are saved. Missing or malformed generator content is reported, not repaired.

The parser accepts JSON or a single JSON fence. Generator answers may also
use Python dictionary literals as in the thesis, including the legacy
`narrative` field; judge deltas must be JSON. Duplicate fields, multiple
fences, malformed edits, negative/fractional/bool node IDs, conflicting edits
and nonfinite values are errors. A node ID written as a JSON string of ASCII
digits (`"22"`) is the integer node 22; any other string is an error. A valid empty delta has explicit empty
lists; `{}` is not a valid extraction. Empty node-change lists can be omitted
by judges, while the three original edge/feature lists are required.

Node IDs are adjacency-matrix positions. `node_matching: position` records
this assumption explicitly. No semantic graph/atom matching is performed;
using DCE's copied graph ID does not establish node correspondence. Added or
removed terminal indices can be described by Reconstruction/Reversal, but
this does not identify entities after a permutation.

Undirected edges are canonical `[min(u,v),max(u,v)]`, including self-loops.
Directed edges retain orientation. Features use their configured names or
column indices. `node_features` in `ProbeNarratives` can explicitly select
columns by name; no derived feature is silently excluded. The selection and
absolute tolerance are saved with the inputs.

`feature_match: identity` retains the thesis's `(node, feature)` scoring.
`transition` compares `(node, feature, from, to)` using exact numeric values.
For a binary feature, 0→1 and 1→0 are therefore distinct. Numeric transitions
are not called feature additions/removals. The TreeCycles pilot selects
`transition`, but its graphs carry no informative node-feature changes;
the choice does not establish a final cross-dataset paper protocol.

## What each probe measures

**Reconstruction** asks the judge to extract only edits mentioned by its
input text, then compares the extraction with the measured true delta.
Scores include precision, recall, F1 and Jaccard for added/removed edges,
feature changes and added/removed nodes, plus the combined tagged-edge set.
Two empty sets score 1; a missing or invented nonempty set scores 0.

**Reversal** performs fresh extraction from direct and inverse answers and
compares the inverse of the direct extraction with the inverse-text
extraction. Feature inversion swaps from/to; edge/node inversion swaps
added/removed. This comparison does not read the true delta or another
probe's results. Separately, the judge receives Alejandra's YES/NO task about
the same entities, opposite edits and consistently swapped classes. Invalid
semantic output is an error even when structural scores are available.
Agreement between two wrong explanations can pass Reversal; Reconstruction
measures their fidelity to the actual graphs.

**Recourse** reuses the existing alternative-edit task, with one JSON
proposal. It always receives the factual graph, original edits, domain and
target class. The no-explanation control receives the same information
except the generator explanation and its associated prompt requirement.
It is a separate arm, not Reconstruction's CTX-off condition.

The application layer validates edge presence, indices, named feature
columns and feature from-values. It preserves the dataset reference, graph
features, surviving edge attributes and weights. Direction is respected.
New unweighted/unattributed edges receive GRETEL's default ones; weighted or
attributed graphs require explicit `edge_defaults.weight`/`features` to add
edges. Recourse does not invent nodes or their attributes. Dataset
`manipulate` runs by default before oracle prediction; it can be disabled
explicitly with `preprocess: false`.

Empty proposals, proposals reusing ANY individual original edit, and proposals
with no measured effect after preprocessing are rejected. The reuse check runs
on both requested and realized edits, in both arms. Adding unrelated edits or
returning a subset of the original delta does not bypass it. Undirected edges
are canonical; directed edges retain orientation and addition/removal is part
of edit identity. Feature reuse means the same node, feature column and numeric
from/to transition, resolving names and column indices to the same column.
A different transition on that column is a different edit. Reused edits and
the rejection phase (`proposal` or `realized`) are saved. A reuse has its own
status, `reused_original`, kept apart from `invalid_proposal`; it counts as an
unsuccessful attempt, and each arm's summary reports `<arm>_n_reused_original`.
`success: target`
requires the saved counterfactual class; `flip` accepts any changed class.
Both flags are retained in output. The prompt requests minimality and domain
compliance; a successful flip alone certifies neither minimality, domain
validity beyond the configured preprocessing, nor the narrative's causal
mechanism. Proposal graphs and realized edits are saved for inspection.

Graph-feature changes and existing-edge value/weight/attribute changes fall outside TypedDelta and are
reported as `unsupported_delta` during generation or `unsupported_edits`
after Recourse preprocessing. This is an explicit supported edit contract,
not a claim of universal coverage of graph transformations.

## Conditions, errors and reproducibility

For Reconstruction/Reversal, `context: on` supplies the factual graph to the
extractor; `off` supplies no additional graph. `mode: dict` selects only the
narrative; `full` selects the actual complete generator response. A full
response can already contain structural information even with CTX-off.
The semantic Reversal judge compares the two selected texts directly.

Records distinguish generation/input errors, malformed generator output
(`unparsed`), backend errors (`model_error`), malformed judge output
(`judge_unparsed`), unsupported edits, invalid proposals and oracle errors.
`success` means evaluation completed, not that a narrative scored perfectly
or a Recourse proposal flipped. `partial_error` retains successful arms and
scores when another component fails. A backend cardinality mismatch yields
an error per attempted prompt; it cannot truncate the sample list silently.

First-pass records save `input_status` separately from `direct_generation` and
`inverse_generation`. Each direction saves backend `call_status`, validated
answer `status`, raw response, prompt and any error. A valid pair with a failed
or malformed answer has overall `partial_error`, not `success`; directions
not generated are `not_attempted`. Backend failures remain `model_error`;
successful calls with malformed or missing structured content are `unparsed`.
Second-pass records retain both directions in `generation_outcomes` and identify
generation failures with `error_origin: generator` and the failed directions.
Reconstruction/Recourse require only a valid direct answer; a failed inverse
does not block them. Reversal requires both answers. Older schema-1 dumps remain
readable: outcomes are derived from saved backend metadata and actual responses.
When the saved call succeeded, answer validity is re-derived from the saved
response with the current parser, so parser changes need no regeneration.
Summaries include generation status counts per direction. Offline manifests
include probe `protocol_version: 6` so results under a changed parser do not
overwrite results from the earlier protocol under the same run identity.

Summaries expose total counts, status counts, number scored, means over valid
scores, and Recourse rates both over all saved attempts and valid proposals.
They do not silently remove malformed outputs. The all-attempt difference
treats errors as unsuccessful; the separate valid rates and raw records
allow a different analysis. Selecting a final paper denominator, paired
sample intersection, statistical treatment or CTX/MODE matrix remains an
analysis decision. The pilot config initially uses only CTX-off/MODE-dict.

First-pass paths distinguish dataset, oracle, explainer, generator/measurement
configuration, run, fold and instance. Complete matrices, one-column node
features, edge attributes, edge weights, graph features and labels are saved.
Dump writes are atomic and failures are fatal. The second pass saves a
manifest with effective judge settings, source paths and SHA256 hashes;
different conditions, folds and source files remain separate. Corrupt dumps
and oracle replay errors are retained as error records. `--fail-on-error`
returns exit 2 after saving results if any attempt failed.

The HuggingFace backend retains the final channel and rejects incomplete
reasoning-only answers. Harmony markers follow the model's
[official chat template](https://huggingface.co/openai/gpt-oss-20b/blob/main/chat_template.jinja).
It also recognizes flattened `analysis...assistantfinalYES` outputs observed
on the cluster, even without an initial `assistantanalysis` role prefix.
Reversal still requires exactly YES or NO in the final answer; extra prose is
an error. The backend also handles Muse's `to=self`/`to=user` framing and
records cached checkpoint revisions.
This parsing check does not replace a live backend smoke test.


## Running

From the repository root. `scripts/slurm_paper_probes.sh` wraps both passes
for a Slurm cluster; it waits for free memory on the assigned GPU
(`MIN_FREE_MB`, `MAX_WAIT`, `WAIT_STEP`) and activates the conda environment
named in the script.

```bash
mkdir -p lab/output/logs
gen_job=$(sbatch --parsable scripts/slurm_paper_probes.sh generate lab/config/probes/paper_treecycles_28nodes_generate.jsonc)
sbatch --dependency="afterok:${gen_job%%;*}" scripts/slurm_paper_probes.sh probe lab/config/probes/paper_treecycles_28nodes_judge.jsonc lab/output/results/paper-probes-tcr-500-28
```

The probe pass reads every saved dump; `SAMPLE_LIMIT=N` restricts it to the
first N. Without Slurm, the second pass is:

```bash
python scripts/run_paper_probes.py --config lab/config/probes/paper_judge_gpt-oss-20b.jsonc \
  --results lab/output/results/paper-probes-pilot --probe reversal --limit 5 --fail-on-error
```

Omit `--probe` to run all configured probes. `--limit` limits saved original
instances; each instance can contain several counterfactuals. Different
judges can evaluate the same dumps without regenerating explanations.
Slurm completion alone does not imply successful parsing or successful
proposals: check the probe `summary.json`.

## Model matrix

`scripts/paper_probe_matrix.py` runs several generators and judges on the
same graph pairs. It freezes the dumps of a finished first pass, checks their
hashes in every later job, and queues one Slurm job per model, chained with
`afterany` so models use one GPU at a time and a failed model does not block
the next. Every generator receives the same saved direct/inverse prompts and
graphs; each judge loads once and evaluates all generators' answers.
`paper_treecycles_10nodes_matrix.json` lists models, checkpoint revisions and
Slurm resources; models load only from cached snapshots under `HF_HOME`.

```bash
python scripts/paper_probe_matrix.py submit --dry-run   # prepare and print the jobs
python scripts/paper_probe_matrix.py submit
python scripts/paper_probe_matrix.py summary --run-root /absolute/path/from/Matrix
python scripts/paper_probe_matrix.py rerun-judges --from-matrix /absolute/path/to/previous/matrix
```

`matrix-summary.csv` compares probe scores and Recourse rates;
`matrix-summary.json` keeps the full summaries and each comparison's
`complete/error/pending` state. `rerun-judges` copies the generators' answers
byte for byte into a new run and queues only the judges.

## Configurations

All under `lab/config/probes/`; each file's header describes its setup.

| Generation | Judge | Setup |
|---|---|---|
| `paper_treecycles_generate.jsonc` | `paper_judge_gpt-oss-20b.jsonc` | First integration run: 50 TreeCycles graphs of 16 nodes |
| `paper_treecycles_10nodes_generate.jsonc` | `paper_treecycles_10nodes_judge.jsonc` | Mini run: 10 graphs of 10 nodes, 5 explained; source of the model matrix |
| `paper_treecycles_28nodes_generate.jsonc` | `paper_treecycles_28nodes_judge.jsonc` | TreeCycles as in the LBS paper: 500 graphs, 28 nodes, DCE + LocalSearch |
| `paper_bbbp_ofs_generate.jsonc` | `paper_bbbp_ofs_judge.jsonc` | BBBP with OFS on the molecule itself, no padding |
| `paper_bbbp_dce_generate.jsonc` | `paper_bbbp_dce_judge.jsonc` | BBBP with the thesis setup (padding, DCE), padding hidden from prompts |
| `paper_aids_dce_generate.jsonc` | `paper_aids_dce_judge.jsonc` | AIDS with the BBBP DCE setup and named feature columns |

`lab/config/snippets/paper_generation_pipeline.json` is the first-pass
pipeline (Qwen3-8B generator). The molecule configs use `SeededMolGenerator`,
`OracleView` (drops bond features and SMILES, which the GCN oracles never
read) and, for AIDS, `FeatureNames`.

## Tests

```bash
python tests/typed_delta_unit.py
python tests/reconstruction_metrics_unit.py
python tests/reconstruction_probe_unit.py
python tests/paper_probes_unit.py
```

The last test runs five graphs from the real seeded TreeCycles generator
through real stages, saving/reloading, factories and TreeCyclesOracle. Its
CF explainer and LLM replies are explicitly scripted test fixtures.
On a machine lacking torch, `--light` isolates only the unused utils package
initializer and Dataset annotation import. It does not emulate oracle
predictions. `--demo-output /absolute/path` retains reviewable input dumps,
proposal graphs and summaries. Those perfect synthetic scores verify the
software; they are not paper results. The tests do not run a real model.
