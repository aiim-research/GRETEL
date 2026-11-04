#!/usr/bin/env python3
"""Generate the selector v4 configs for PROTEINS from the v3b-pre-frozen config.

v4 keeps the LBS skeleton and the v3b selector (P0 on the ranked sweeps, flip target, tabu, sampled
swap/reduce at sample_temp 4, offline-pretrained and frozen) and adds two selector-side changes:

* ``evidence_features``: per candidate pair, what the search already observed on this graph (moves
  evaluated with the pair, mean and max counterfactual probability, its mean against the instance mean,
  flip rate). Offline, every pretrained ranker sat at a held-out AUC of about 0.70 with the structural
  features alone, whatever the model; this is information they did not have. Needs a checkpoint
  pretrained with the evidence block (``v4-collect`` examples, ``scripts/pretrain_selector_v3.py``).
* ``adaptive_sweep_temp``: in a sweep the temperature rises from 0.25 towards 1 as the state's
  candidates are tried and fail, to give back some of the GED that v3b-pre-frozen gives up.

Arms (seeds 0, 1, 2; the v3b-pre-frozen seeds are the reference):

    v4         evidence + adaptive temperature
    v4-ev      evidence only
    v4-adapt   adaptive temperature only (the v3b pretrained checkpoint)

Round 2, after v4 = v4-ev to the last digit on seed 0: the share-of-tried temperature rule never moved
(a retry only tries about two candidates before its phase share runs out), and the evidence cost GED
without saving calls, likely because the network learned to read the evidence on uniform-proposal data
while online the evidence accumulates under its own ranking. So:

    v4-adapt2  temperature doubling per failed sweep on the same state (sweep_temp_growth 2)
    v4-ev2     evidence, pretrained on the uniform collection plus an on-policy one
    v4-onpolicy-collect  the v3b-pre-frozen policy on the 72 collection instances, logging examples

    python scripts/gen_selector_v4_configs.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DST = REPO / "lab/config/generate_minimize/proteins40/dce"
SRC = DST / "dce-lcls-net-v3b-pre-frozen/generate_minimize0.jsonc"
PRE_V4 = "models/pretrained/edge_selector_v4-pre_PROTEINS.pt"
PRE_V4B = "models/pretrained/edge_selector_v4-pre2_PROTEINS.pt"
COLLECT_IDS = REPO / "lab/config/snippets/proteins_72_collect_instances.json"
ARMS = {
    "v4": {"init_checkpoint": PRE_V4, "selector": {"evidence_features": True, "adaptive_sweep_temp": True}},
    "v4-ev": {"init_checkpoint": PRE_V4, "selector": {"evidence_features": True}},
    "v4-adapt": {"selector": {"adaptive_sweep_temp": True}},
    "v4-adapt2": {"selector": {"sweep_temp_growth": 2.0}},
    "v4-ev2": {"init_checkpoint": PRE_V4B, "selector": {"evidence_features": True}},
}
SEEDS = (0, 1, 2)


def main() -> int:
    for arm, extra in ARMS.items():
        for sd in SEEDS:
            conf = json.load(open(SRC))
            name = arm if sd == 0 else f"{arm}-s{sd}"
            conf["experiment"]["scope"] = f"proteins40_dce_lcls-net-{name}"
            par = conf["doe-triplets"][0]["explainer"]["parameters"]["minimizer"]["parameters"]
            par["seed"] = sd
            par["model_tag"] = name
            if "init_checkpoint" in extra:
                par["init_checkpoint"] = extra["init_checkpoint"]
            par["selector"] = {**par["selector"], **extra["selector"]}
            out = DST / f"dce-lcls-net-{name}" / "generate_minimize0.jsonc"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(conf, indent=4) + "\n")
            print(out.relative_to(REPO), conf["experiment"]["scope"])
    # on-policy collection: the v3b-pre-frozen policy itself, on the instances outside the evaluation set
    conf = json.load(open(SRC))
    conf["experiment"]["scope"] = "proteins72collect_dce_lcls-net-v4-onpolicy-collect"
    conf["evaluator"]["parameters"]["instance_ids"] = [r["id"] for r in json.load(open(COLLECT_IDS))["instances"]]
    par = conf["doe-triplets"][0]["explainer"]["parameters"]["minimizer"]["parameters"]
    par["model_tag"] = "v4-onpolicy-collect"
    par["collect_examples"] = "lab/output/selector_pretrain/proteins_v4_onpolicy_collect.pt"
    out = DST / "dce-lcls-net-v4-onpolicy-collect" / "generate_minimize0.jsonc"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(conf, indent=4) + "\n")
    print(out.relative_to(REPO), conf["experiment"]["scope"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
