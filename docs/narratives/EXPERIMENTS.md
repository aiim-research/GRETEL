# Narrative probes paper: experiment plan

Plan of every run needed for the paper "What Changed and Why: Probing Graph Counterfactual Narratives and Their Narrators", written on 1 November 2026. It is the source for building the experimental setup on `dev-narrative`. Configs do not need to exist yet: this file says what has to be run, and section 3 says what code is missing to run it.

The paper drafts live in the phd project (`2-2026-09-What Changed and Why - .../article-es/` and `.../article/`). Their experimental section is written as one `\TODO` per experiment, and the research questions below (RQ1 to RQ9) are the ones in those drafts. This file supersedes the September `PLAN-EXPERIMENTOS.md` of the phd project for anything about runs.

## 0. Starting point

**Branch.** `dev-narrative` starts from `origin/main` at `ca7149b27` (the merge of #87, the BLS fixes). The intended order is:

1. Merge Gianrico's PRs into this branch: #88 (typed edit set and reconstruction metrics), #89 (seeded Tree-Cycles and molecule generators), #90 (AIDS class names) and then #91 (graph probes, which contains #88 and #89 underneath).
2. Cherry-pick from `dev-bls-selector` whatever the minimality arm needs. Candidates to check: the Tree-Cycles generator fixes (`0f3bb6a97`) and any BLS change not already in #87.
3. Build the missing code of section 3 on top.

**What #91 already provides**, so it does not have to be written again:

| Piece | Where (in #91) |
|---|---|
| Typed edit set as ground truth, strict precision, recall, F1 and Jaccard per edit type | `src/utils/typed_delta.py`, `src/utils/reconstruction_metrics.py` |
| Two passes: generation (`ProbeNarratives`, `ProbeDump`) and offline judges (`scripts/run_paper_probes.py`) | `src/evaluation/future/stages/probe_*.py` |
| Two-part generator answer (`edits` plus `Natural_Language_Explanation`), strict parsing, error statuses | `docs/paper-probes.md` |
| Judge configured apart from the generator, with a family check | `probe_stage.py`, offline configs |
| Reconstruction with `context` on/off and `mode` dict/full | `src/utils/reconstruction_probe.py` |
| Recourse with the no-explanation control, reuse rejection, `target` and `flip` success | `src/utils/recourse_probe.py` |
| Reversal (structural plus Alejandra's YES/NO) | `src/utils/reversal_probe.py`, to be replaced, see C1 |
| HuggingFace backend (transformers or vLLM) and OpenRouter backend (untested) | `src/LLMexplaneability/huggingface.py`, `openrouter.py` |
| `OracleView` (drops bond features and SMILES) and `FeatureNames` (AIDS atom names) | `src/dataset/manipulators/` |
| Padding hidden from prompts (BBBP DCE config) | `lab/config/probes/paper_bbbp_dce_*.jsonc` |
| Model matrix runner and Slurm script | `scripts/paper_probe_matrix.py`, `scripts/slurm_paper_probes.sh` |

## 1. Design the runs follow

These come from the paper drafts and are fixed unless stated in section 2.

- **Three probes.** Reconstruction measures R1 (the edit set can be recovered from the narrative). Recourse measures R2 (the stated mechanism suffices to flip the oracle along another path), with its no-explanation control. Consistency measures C, a property of the **narrator** (the LLM that writes the narrative), not of a single narrative.
- **Consistency** has three tests, each scored only by an external judge with a **compatibility** criterion (both mechanisms could be true of the same change). A narrative with no mechanism fails.
  - Inversion: narrate G' to G. The mechanism must be compatible with the negation of the original.
  - Renaming: permute the node indices of G and G' and narrate again. The mechanism must be compatible with the original, with the names changed.
  - Identity: narrate the same input again with another sampling seed. The mechanism must be compatible with the original. This is the noise floor: inversion and renaming are reported as a drop with respect to it.
- **No structural scoring for Consistency.** It was almost redundant with Reconstruction and did not measure the mechanism.
- **Graph level only.** Tree-Cycles, BBBP and AIDS. No node-level datasets.
- **Minimality factor.** Every instance is narrated from its DCE seed and from its LBS-minimized version (not necessarily a subset of the seed). A third arm, a random subset of the seed of the minimized size, controls for size and is scored with Reconstruction only.
- **Surface metrics** (distinct-n, MTLD, Self-BLEU) are a negative control, not quality measures.
- **The reader** is an expert in the domain of the data. Nodes are serialized with named attributes, without padding nodes, 2D coordinates or bond types.
- **Out of scope for this paper:** node-level runs, the structural Reversal score, minimization as a consistency test (by the reachability proposition of the LBS paper the minimized counterfactual can flip the model for another reason), the veracity control (phd line B4), background perturbation as a consistency test, and the sampling estimator of sufficiency (phd line B3).

## 2. Decisions to take before writing configs

| # | Decision | Status and recommendation |
|---|---|---|
| D1 | Datasets | Tree-Cycles, BBBP and AIDS, agreed with Gianrico. |
| D2 | Tree-Cycles size | With 28 nodes all probes saturate (Gianrico's pilot). Use a harder variant (more nodes and cycles, e.g. the existing 64-node setting) or keep Tree-Cycles only as a validation with a known mechanism. |
| D3 | Padding | Either retrain the oracles without padding or keep the current oracles and only hide padding from the prompts (already done in #91 for BBBP DCE). The second option keeps the LBS paper's setup. |
| D4 | Seed generator | DCE. Its seeds are real molecules and LBS minimizes them. OFS can produce chemically invalid structures. |
| D5 | Primary Reconstruction condition | Pending. The paper's inherited text says narrative only and no context, but the expert reader has the instance at hand, which is closer to CTX-on. |
| D6 | Feature matching | `identity` or `transition`, and whether a judge answer missing a from/to value drops the whole instance (`judge_unparsed`). Open protocol question of #91, matters for BBBP and AIDS. |
| D7 | Soft matching | Whether it is still needed with named features (the judge outputs column names). If kept: lemmatization plus embedding similarity with threshold tau for attribute values, exact matching for node indices. |
| D8 | Narrators | The paper lists Qwen2.5-Instruct 3B, 7B, 14B and 32B, Llama-3.1-8B-Instruct and Gemini 2.5 Flash. #91 pilots with Qwen3-8B. Fix the list and the exact revisions. |
| D9 | Judges | DeepSeek-V3.2 (primary), GPT-OSS-120B and Gemini 3 Flash (robustness), never on narratives of their own family. Access through OpenRouter needs a live smoke test. |
| D10 | Sampling | Temperature T > 0 (the identity test needs non-deterministic sampling), top-p, max tokens and seeds, the same for generation and identity. |
| D11 | Sizes | N instances per dataset (the paper says 100), the default Recourse budget K and the values of the budget sweep. |

## 3. Code to write on top of #91

| # | What | Notes |
|---|---|---|
| C1 | **Consistency probe** replacing Reversal in the paper path | One judge verdict per test (inversion, renaming, identity). New judge prompt with the compatibility criterion: compatible with the negation for inversion, with the original for renaming and identity, negative when no mechanism is stated. The current YES/NO prompt checks same entities, inverted edits and swapped classes, which is mostly structural. |
| C2 | **Renaming transformation** in `ProbeNarratives` | Draw a node permutation pi per instance (seeded, saved), apply it to G and G' (features, adjacency, edit set), narrate the permuted pair, and keep pi so the judge can compare mechanisms with the names mapped back. |
| C3 | **Identity transformation** in `ProbeNarratives` | A second direct generation with another sampling seed, same prompt. Requires per-call seed control in the backends (C10). |
| C4 | Drop the structural Reversal score from the paper path | Keep the code if useful elsewhere, but no paper table uses it. |
| C5 | **Minimality arms paired by instance** | Seed arm (`Dummy` minimizer, as in `lab/config/llm_exp_generate_minimize/*/dce/dce-dummy`) and minimized arm (`LocalSearch`) on the same instances and oracle, with a pairing key in the dumps. |
| C6 | **Random-subset control arm** | For each instance, draw a random subset of the seed's edits with the size of the minimized edit set (seeded, saved), build G plus that subset, narrate it and score it with Reconstruction only. It is not a valid counterfactual, so Recourse and the flip check do not apply. |
| C7 | **Recourse budget K** | K as an explicit prompt parameter (the old prompt has a `K={{K}}` placeholder) and enforced on the proposal, the same K in both arms, plus a runner for the sweep. |
| C8 | **Soft matcher**, if D7 keeps it | Run offline over saved extractions, reported next to the strict score. |
| C9 | **Surface metrics** offline | `DistinctN`, `MTLD` and `SelfBLEU` already exist in `src/LLMexplaneability/explanation_evaluator.py`. Run them over the narrative field of every saved answer. |
| C10 | **Backends** | Per-call temperature, top-p, max tokens and seed for every backend, saved in the dumps. A Gemini backend in the new interface, or Gemini through OpenRouter. Live smoke test of OpenRouter with the three judges. |
| C11 | **Self-evaluation mode** | An explicit flag that lets the narrator judge its own inversion test, only for E6. The family check must stay on everywhere else. |
| C12 | **Analysis scripts** | See section 6. |

## 4. Experiments

Every experiment reads the same generated pairs, and nothing is generated twice. "Pair" means one instance, one arm and one narrator.

| Id | What | Datasets and arms | Who | Feeds |
|---|---|---|---|---|
| **E0** | Pilot after C1 to C3, C10: one dataset, a few instances, one narrator, primary judge, every probe. Checks parse rates, statuses and cost per pair before the full matrix. Includes the Tree-Cycles size choice (D2). | Tree-Cycles (chosen variant), then BBBP | 1 narrator, primary judge | sanity |
| **E1** | **Generation pass.** Direct, inverse, renamed and identity narratives for every pair, with the two-part answer. | 3 datasets x 2 arms (seed, minimized) x N instances | all narrators | everything |
| **E2** | **Primary judge pass** over E1: Reconstruction under the four conditions (CTX on/off x MODE dict/full), Recourse with and without explanation at the default K, the three Consistency verdicts. | all of E1 | primary judge | RQ1 to RQ7 |
| **E3** | **Minimality control arm.** Narrate the random-subset pair and score it with Reconstruction. | 3 datasets x random-subset arm x N | all narrators, primary judge | RQ5 |
| **E4** | **Recourse budget sweep.** With and without explanation for every K in the sweep. | a subset, e.g. Tree-Cycles and BBBP, minimized arm | primary judge as the acting model | RQ9 |
| **E5** | **Judge robustness.** Rerun the judges over E1 without regenerating (`rerun-judges`): Reconstruction (the four conditions on the RQ7 subset, CTX-off/MODE-dict everywhere), Recourse proposals and Consistency verdicts. | all, or a stated subset | GPT-OSS-120B, Gemini 3 Flash | robustness block, RQ7 |
| **E6** | **Self-evaluation.** Each narrator judges its own inversion test, compared with the external primary judge. Repeats Alejandra's preliminary result (Llama-70B and Qwen-72B at 1.0 with self-judging, 0.83 and 0.88 with Gemini) under the compatibility criterion. | Tree-Cycles, optionally BBBP | narrators as judges (C11) | robustness block |
| **E7** | **Surface metrics** over every narrative of E1 and their correlation with the three probes. | offline | none | surface-control block |
| **E8** | **Strict versus soft matching** over the saved extractions of E2 and E5. | offline | none | RQ8 |

## 5. Calls per pair

With the primary judge: 4 generations (direct, inverse, renamed, identity), 4 Reconstruction extractions (CTX x MODE), 3 Consistency verdicts and 2 Recourse proposals, i.e. **13 calls per pair**. On top of that:

| Extra | Calls |
|---|---|
| E3, random-subset arm | 1 generation plus the Reconstruction extractions per instance and narrator |
| E4, budget sweep | 2 per value of K per pair of the subset |
| E5, each robustness judge | 9 per pair (4 extractions, 3 verdicts, 2 proposals), fewer if Reconstruction runs only CTX-off/MODE-dict |
| E6, self-evaluation | 1 per pair of the subset |

Example: 3 datasets x 100 instances x 2 arms x 6 narrators = 3,600 pairs, so about 47,000 primary-judge calls, plus up to about 65,000 for the two robustness judges if they cover everything. Restricting E5 to a subset is the main lever on cost.

## 6. Analyses (one script each, over the saved records)

| Analysis | RQ |
|---|---|
| Reconstruction gap MODE-full minus MODE-dict on the intersection of samples across conditions, per dataset and narrator, under strict and soft matching | RQ1 |
| Recourse flip rate with and without explanation, lift, rejected proposals (`reused_original`, invalid), and on Tree-Cycles the fraction of flipping proposals that break or create a cycle (the known mechanism) | RQ2 |
| Consistency per test and narrator, drops of inversion and renaming with respect to identity, fraction of instances passing all three | RQ3 |
| Pairwise correlation between the three probes, Reconstruction against Recourse quadrants, systematic failures per narrator, agreement between the two identity samples | RQ4 |
| Seed versus minimized on the three probes, paired by instance, and the random-subset arm on Reconstruction | RQ5 |
| Edit family (edges versus atom attributes on BBBP and AIDS), best-of over conditions (ceiling), narrator scale, dataset | RQ6 |
| The CTX x MODE matrix under the three judges | RQ7 |
| Strict versus soft matching per edit family | RQ8 |
| Recourse lift as a function of K | RQ9 |
| Judge agreement: correlation and MAE of Reconstruction F1, change of Recourse lift by acting model, kappa of Consistency verdicts, and self-evaluation versus external judge | robustness block |
| Correlation of the surface metrics with each probe | surface-control block |
| Parse rates and status counts per narrator, as a first-class result | setup and appendix |

Comparisons across conditions always use the intersection of samples valid in every condition, and every table reports N per cell.

## 7. Suggested order

1. Take the decisions D2 to D11 that block configs (at least D2, D3, D6, D8, D9, D10).
2. Merge #88 to #91 and the cherry-picks, then write C1 to C3, C5 and C10.
3. E0 on the chosen Tree-Cycles variant, then on BBBP.
4. C6, C7, C9, C11 and C12, then E1 and E2 for one narrator end to end.
5. The full E1 and E2, then E3 to E6.
6. E7 and E8 offline, and the analyses of section 6.

## 8. Status on 3 November 2026

Step 2 of section 0 is done: Gianrico's #88 and #91 are merged into this branch, #90 went to `main` (#92 brought the general framework fixes of `dev-bls-selector` there too), and #89 was closed because `main` fixed `TreeCyclesRand` and `MolGenerator` in place; the probe configs use them.

Decisions taken:

| # | Decision |
|---|---|
| D2 | Tree-Cycles with 500 graphs of 64 nodes, 40 % of the nodes in cycles, rule oracle. |
| D3 | The current oracles, padding hidden from the prompts (BBBP and AIDS as in Gianrico's DCE pilots). |
| D4 | DCE seeds, LBS with the LBS paper's parameters. |
| D8 | Qwen2.5-Instruct 3B, 7B, 14B and 32B and Llama-3.1-8B-Instruct in bf16. Gemini 2.5 Flash waits for API access. |
| D9 | Until there is API access, GPT-OSS-120B (local) is the only judge and plays the primary role; DeepSeek-V3.2 and Gemini 3 Flash, and with them E5, wait. |
| D10 | Temperature 0.7, top-p 0.95, 1024 output tokens, seed 0; the identity narrative uses seed 1. |
| D11 | 100 instances per dataset (fold 0 of 5 splits for Tree-Cycles, of 20 for BBBP and AIDS); Recourse K = the instance's minimized edit count, the same for both arms and for the seed and minimized pairs of an instance; sweep K in 1, 2, 3, 5, 8. |

Still open: D5 (all four CTX x MODE conditions are run, so it only changes which one the tables lead with), D6 (runs use `transition`, as the pilots), D7 (C8 not written). Also open: the Tree-Cycles domain text (`dataset_infos/tcr.json`) states the classification mechanism to the narrator, which may explain part of the saturation of the pilots.

Code: C1 (`src/utils/consistency_probe.py`), C2 and C3 (`directions` of `ProbeNarratives`), C4 (Reversal stays in the code, no paper config uses it), C5 and C6 (`arms`), C7 (`budget` of the Recourse conditions), C10 for the HuggingFace backend (per-prompt seeds, sampling saved per answer) and C11 (`self_evaluation`) are written and tested (`tests/narrative_arms_unit.py`). Not written: C8, C9 runs and C12 (offline, after the runs), and API backends with per-call seeds (OpenRouter refuses per-prompt seeds for now). The runs, their batches and the server steps are in `RUNS.md`.

## 9. Update of 4 November 2026: protocol v2

The padding turned out to be a framework-wide problem (task 0 of the research plan, `docs/v2/README.md`): the GCN oracles read the amount of padding, and the minimizers edited padding rows (a quarter of the BBBP minimized arm in the first smoke). The pairs pass now runs on protocol v2, which replaces D3:

- **No padding.** Every graph at its own size, the oracles retrained without padding (and with the GCN layers their configs ask for). OracleView and FeatureNames stay. Nothing is hidden from the prompts any more.
- **Isolated nodes deleted.** A node an edit leaves without edges is deleted; salt ions are removed at load. Deleted atoms are reported as `nodes_removed` and left out of the counterfactual's text.
- **Seed arm.** DCE's nearest molecule of the other class, matched atom to atom by assignment and written as edits on the input (bond edits, atom substitutions, deleted atoms). It is always a counterfactual. With the assignment, the seed is far more readable than in v1 (BBBP: about 18 bond edits and 45 feature changes, against about 115 index-matched edits).
- **Minimized arm.** BLS-F, the bounded local search over bond edits and atom substitutions towards the seed (`local_search_features`), not the first paper's edge-only BLS: with seeds that change atoms, the edge-only BLS often cannot improve and returns the seed, which would erase the minimality contrast. The paper should name it as the LBS variant with feature substitutions. BBBP: about 2.3 bond edits and 7.5 atom substitutions.
- **Smokes.** BBBP 100 of 100 and AIDS 98 of 98 instances with the three arms valid, Tree-Cycles 64 all valid, and an E0 matrix prepares from these pairs. The configs and the tutorial (`RUNS.md`) are updated; the scopes are `narr-<dataset>_dce_lclsf`.
- **Judges and prompts (4 November).** Until there is API access every judge is local: GPT-OSS-120B is the primary judge of the three probes and of the Recourse sweep, and GLM-4.7-Flash (another family, already used as a judge in Gianrico's pilots) is a local robustness judge on Reconstruction CTX-off/MODE-dict, Consistency and Recourse, a first part of E5. The Recourse prompt (Alejandra's FlipRateEvaluator task) no longer carries the unfilled `K={{K}}` placeholder: the goal states the budget K, and the original counterfactual is named as the judge sees it (ORIGINAL EDITS). The matrix runs on any `fast` GPU of the `cuda` partition, the jobs of one role side by side.
- **Prompts (4 November, protocol 9).** Gianrico's prompts (the Reconstruction extractor, the two-part output schema, the graph text) are unchanged. The narrator's rules (the explanation pipeline's) ask for the root causes the modifications support, at most five, instead of "the top 3-5": three causes for a one-edit counterfactual invite invented ones and would bias the minimality factor. The Consistency judge prompt (new) was refined and checked for format with a small model. Open: the narrator prompts are long (median 8 to 9 thousand tokens; 4 % of BBBP and 11 % of AIDS above 16 384, a few up to 35 000), mostly redundant edge lists on Tree-Cycles and raw 23- to 42-column feature vectors on the molecules.
- **Runs (5 November).** Two accounts in parallel (Rodrigo: Qwen2.5-32B and Llama-3.1-8B; Gianrico: Qwen2.5-3B, 7B and 14B), same pairs, judges, revisions and vLLM version; one run per dataset and account, one batch per dataset (`paper_probe_matrix.py fleet`, throttled Slurm arrays), each launched when the previous one is over: Tree-Cycles, BBBP, AIDS. The separate E0 pilot is dropped: Tree-Cycles, which ends first, plays its role. The judge validation V0 (annotated and synthetic narratives, three candidate judges) is prepared by Rodrigo and Gianrico together after the batches; if it disqualifies the primary judge on a probe, only the judges are rerun. Tutorials: `RUNS.md`, `RUNS_RODRIGO.md`, `RUNS_GIANRICO.md`.
