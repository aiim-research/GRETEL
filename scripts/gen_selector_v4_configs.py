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

    python scripts/gen_selector_v4_configs.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DST = REPO / "lab/config/generate_minimize/proteins40/dce"
SRC = DST / "dce-lcls-net-v3b-pre-frozen/generate_minimize0.jsonc"
PRE_V4 = "models/pretrained/edge_selector_v4-pre_PROTEINS.pt"
ARMS = {
    "v4": {"init_checkpoint": PRE_V4, "selector": {"evidence_features": True, "adaptive_sweep_temp": True}},
    "v4-ev": {"init_checkpoint": PRE_V4, "selector": {"evidence_features": True}},
    "v4-adapt": {"selector": {"adaptive_sweep_temp": True}},
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
