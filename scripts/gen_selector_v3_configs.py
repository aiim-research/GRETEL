#!/usr/bin/env python3
"""Generate the selector v3 configs for PROTEINS from the proteins40 v2e config.

v3 = v2e + Thompson-sampled node-feature methods (at most ``method_max`` per candidate) + no swap
phase + optional offline pretraining. Arms on the 40 evaluation instances:

    v3-random      uniform proposals (no model)
    v3-frozen      untrained seeded network, never updated
    v3-scratch     online learning from the untrained network
    v3-pre-frozen  offline-pretrained network, never updated
    v3             offline-pretrained network plus online learning

plus ``v3-collect``: uniform proposals on the fold-0 test instances outside the evaluation set, dumping
every scored example for the offline pretraining (``scripts/pretrain_selector_v3.py``).

    python scripts/gen_selector_v3_configs.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v2e/generate_minimize0.jsonc"
DST = REPO / "lab/config/generate_minimize/proteins40/dce"
EVAL_IDS = REPO / "lab/config/snippets/proteins_40_instances.json"
COLLECT_IDS = REPO / "lab/config/snippets/proteins_72_collect_instances.json"
PRETRAINED = "models/pretrained/edge_selector_v3-pre_PROTEINS.pt"
COLLECT_FILE = "lab/output/selector_pretrain/proteins_v3_collect.pt"

V3 = {"method_policy": "thompson", "method_max": 2, "skip_swap": True}
ARMS = {
    "v3-random": {"selector_mode": "random"},
    "v3-frozen": {"selector_mode": "frozen"},
    "v3-scratch": {"selector_mode": "learned"},
    "v3-pre-frozen": {"selector_mode": "frozen", "init_checkpoint": PRETRAINED},
    "v3": {"selector_mode": "learned", "init_checkpoint": PRETRAINED},
}


def make(arm: str, extra: dict, ids: list[int], scope: str) -> dict:
    conf = json.load(open(SRC))
    conf["experiment"]["scope"] = scope
    conf["evaluator"]["parameters"]["instance_ids"] = ids
    par = conf["doe-triplets"][0]["explainer"]["parameters"]["minimizer"]["parameters"]
    par.update(V3)
    par.update(extra)
    par["model_tag"] = arm
    return conf


def main() -> int:
    eval_ids = [r["id"] for r in json.load(open(EVAL_IDS))["instances"]]
    collect_ids = [r["id"] for r in json.load(open(COLLECT_IDS))["instances"]]
    assert not set(eval_ids) & set(collect_ids)
    jobs = {f"dce-lcls-net-{arm}": make(arm, extra, eval_ids, f"proteins40_dce_lcls-net-{arm}")
            for arm, extra in ARMS.items()}
    jobs["dce-lcls-net-v3-collect"] = make("v3-collect", {"selector_mode": "random", "collect_examples": COLLECT_FILE},
                                           collect_ids, "proteins72collect_dce_lcls-net-v3-collect")
    for name, conf in jobs.items():
        out = DST / name / "generate_minimize0.jsonc"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(conf, indent=4) + "\n")
        print(out.relative_to(REPO), conf["experiment"]["scope"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
