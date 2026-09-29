# `llm_exp_generate_minimize/` - narratives over generated counterfactuals

Configurations that run the LLM pipeline (`snippets/minimizing_llm_pipeline.json`)
on top of a generate-and-minimize explainer, so that every counterfactual is
verbalised and the resulting narrative is probed.

## The ICLR study

`tcr-500-28/`, `bbbp/` and `aids/` are the graph-level arm of the ICLR paper on
probes for counterfactual narratives. Each dataset has two arms over **the same
instances and the same oracle**:

| Arm | Minimizer | What it is |
|---|---|---|
| `dce/dce-dummy` | `meta.minimizer.dummy.Dummy` | the redundant seed counterfactual DCE returns |
| `dce/dce-lcls` | `metaheuristic.local_search.LocalSearch` | the same seed after LBS minimisation |

The contrast between the two arms is the **minimality factor**: the only thing
that varies is the size of the edit set the narrative has to describe. Three
folds each, matching the three-seed stability convention of the LBS paper.

`tcr-500-28` uses the custom Tree-Cycles oracle, so the true decision rule is
known by construction and the stated mechanism can be checked against it rather
than only for internal consistency. `bbbp` and `aids` use the GCN oracle.

Each triplet composes a `dataset_infos/` snippet, which is what supplies the
`domain` block of the generation prompt. `tcr.json`, `bbbp.json` and `aids.json`
were written for this study.

## Running one

```
export GEMINI_API_KEY=...        # or GEMINI_API_KEYS=k1,k2,... to rotate
python main.py lab/config/llm_exp_generate_minimize/tcr-500-28/dce/dce-lcls/generate_minimize0.jsonc
```

Narratives are dumped under `lab/llm_explanations/`, metrics under
`lab/output/`.

## What the pipeline measures today, and what it does not

`minimizing_llm_pipeline.json` runs `LLMexplanation` (direct and inverse
narrative), `LLMexplanationFlipRate` and
`LLMexplanationContrastiveExplanation`.

Two caveats that matter for the paper and are **not** fixed by these configs:

1. **The judge is the generator.** Both metric stages call
   `explanation.context.llm`, the same model that wrote the narrative. The paper
   argues explicitly against self-judging, so these numbers are a preliminary
   signal and have to be re-run once a separate judge is wired in.
2. **The flip rate has no control.** There is no matched no-explanation arm, so
   the current number conflates what the narrative contributed with what the
   model could have guessed from the graph.

`docs/` in the phd project tracks both as open work.
