#!/usr/bin/env python3
"""One-fold preliminary of the selector arms on the other datasets (does training matter anywhere else?).

On PROTEINS (10 folds) the untrained network is as good as uniform proposals (GED p 0.73), and the trained
selector beats both only where single removals work (GED -13 %, calls -17 % on those instances). This
repeats the four arms of ``gen_selector_proteins_folds.py`` on the test split of fold 0 of every dataset
with a committed cache, each in its own tree ``lab/config/generate_minimize/<dataset>-sel/dce/``:

    dce-lcls                 plain LBS (the dataset's thesis config plus strict_budget, seed 0, budget 10000)
    dce-lcls-sel             the selector, pretrained on this dataset (see below) and frozen
    dce-lcls-sel-untrained   the same network and policy with untrained weights
    dce-lcls-sel-random      the same skeleton with uniform proposals
    dce-lcls-sel-collect     uniform proposals on the test split of FOLD 1, dumping every scored example

The minimizer settings are the PROTEINS ones; only the flags the dataset's own thesis config sets
differently (``attributed``, ``recompute_features``, neighbourhood sizes) are carried over, so the
oracle is queried exactly as in the thesis. Everything else in the config (dataset, oracle, padding,
generator, evaluator) is the dataset's thesis config.

Pretraining follows the PROTEINS recipe with one difference: the examples come from the fold-1 test
instances, which are disjoint from the evaluated fold-0 test split (on PROTEINS they were fold-0 test
instances outside the 40-instance evaluation set, so fold 0 of the 10-fold run contains its own
pretraining instances; here nothing is evaluated on an instance the selector learned from). Order:

    python scripts/run_revision_queue.py --list docs/selector/DATASETS_FOLD0_1-collect.md --workers 2
    for ds in <datasets>; do python scripts/pretrain_selector_v3.py \\
        --examples lab/output/selector_pretrain/${ds}_collect.pt \\
        --config lab/config/generate_minimize/${ds}-sel/dce/dce-lcls-sel/generate_minimize0.jsonc \\
        --out models/pretrained/edge_selector_sel-pre_${ds}.pt --report models/pretrained/edge_selector_sel-pre_${ds}.json; done
    python scripts/run_revision_queue.py --list docs/selector/DATASETS_FOLD0_2-selector.md --workers 2

    python scripts/gen_selector_datasets_fold0.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TREE = REPO / "lab/config/generate_minimize"
SRC = TREE / "proteins-sel/dce"
DOCS = REPO / "docs/selector"
# ordered by the cost of a fold (test instances x size); colors-3 has 1050 test instances per fold
DATASETS = ["asd", "tcr-500-28", "tcr-tco-300", "synthie", "bbbp", "aids", "colors-3"]
ARMS = ["dce-lcls", "dce-lcls-sel", "dce-lcls-sel-untrained", "dce-lcls-sel-random", "dce-lcls-sel-collect"]
# flags the thesis config of the dataset decides (how the oracle is queried), kept from it
DATASET_FLAGS = ["attributed", "recompute_features", "neigh_factor", "runtime_factor", "max_runtime", "max_neigh"]
BUDGET = 10000
COLLECT_BUDGET = 2000          # as the PROTEINS collection
CACHE = "lab/data/cache/generator_outputs"
EVAL_FOLD, COLLECT_FOLD = 0, 1


def set_fold(conf: dict, fold: int) -> None:
    for prop in conf["experiment"]["parameters"]["propagate"]:
        if prop["in_sections"] == ["doe-triplets/explainer"]:
            prop["params"]["fold_id"] = fold
    conf["doe-triplets"][0]["explainer"]["parameters"]["fold_id"] = fold


def main() -> int:
    queues = {"1-collect": [], "2-selector": []}
    for ds in DATASETS:
        thesis = json.load(open(TREE / ds / "dce/dce-lcls/generate_minimize0.jsonc"))
        thesis_min = thesis["doe-triplets"][0]["explainer"]["parameters"]["minimizer"]["parameters"]
        for arm in ARMS:
            src_arm = "dce-lcls-sel-random" if arm == "dce-lcls-sel-collect" else arm
            src = json.load(open(SRC / src_arm / "generate_minimize0.jsonc"))
            conf = json.loads(json.dumps(thesis))
            fold = COLLECT_FOLD if arm == "dce-lcls-sel-collect" else EVAL_FOLD
            set_fold(conf, fold)
            conf["experiment"]["scope"] = f"{ds}-sel_" + arm.replace("dce-lcls", "dce_lcls", 1)
            conf["evaluator"]["parameters"].pop("instance_ids", None)
            expl = conf["doe-triplets"][0]["explainer"]["parameters"]
            expl["generator_cache"] = CACHE
            expl["minimizer"] = json.loads(json.dumps(src["doe-triplets"][0]["explainer"]["parameters"]["minimizer"]))
            par = expl["minimizer"]["parameters"]
            for flag in DATASET_FLAGS:
                if flag in thesis_min:
                    par[flag] = thesis_min[flag]
                else:
                    par.pop(flag, None)
            par["seed"] = 0
            par["max_oracle_calls"] = COLLECT_BUDGET if arm == "dce-lcls-sel-collect" else BUDGET
            if arm == "dce-lcls":
                par["strict_budget"] = True
            if arm == "dce-lcls-sel":
                par["init_checkpoint"] = f"models/pretrained/edge_selector_sel-pre_{ds}.pt"
            if arm == "dce-lcls-sel-collect":
                par["model_tag"] = "selector-collect"
                par["collect_examples"] = f"lab/output/selector_pretrain/{ds}_collect.pt"
            out = TREE / f"{ds}-sel/dce" / arm / f"generate_minimize{fold}.jsonc"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(conf, indent=4) + "\n")
            queues["2-selector" if arm == "dce-lcls-sel" else "1-collect"].append(str(out.relative_to(REPO)))
    for name, cfgs in queues.items():
        what = ("the collection runs (fold 1) and the arms that need no checkpoint (fold 0)" if name == "1-collect"
                else "the pretrained selector (fold 0), after scripts/pretrain_selector_v3.py on each collection")
        (DOCS / f"DATASETS_FOLD0_{name}.md").write_text(
            f"# Selector arms on the other datasets, one fold: {what}\n\n"
            "Generated by `scripts/gen_selector_datasets_fold0.py` (its docstring gives the order). "
            "Auto-managed by `scripts/run_revision_queue.py --list <this file>`.\n\n"
            + "\n".join(f"- [ ] {c}" for c in cfgs) + "\n")
        print(f"{name}: {len(cfgs)} configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
