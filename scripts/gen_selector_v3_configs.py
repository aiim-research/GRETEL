#!/usr/bin/env python3
"""Generate the selector v3 configs for PROTEINS from the proteins40 v2e config.

v3 = v2e with the same LBS skeleton (swap included, every node-feature method evaluated) and changes on
the selector side only, chosen with ``scripts/selector_offline_bench.py`` on the v3-collect examples:

* P0, proposal policy: scores standardised inside each candidate set before the Plackett-Luce sample,
  removal temperature 0.25 (the raw logits spread far less than the Gumbel noise, so the v2e policy
  tried an almost random order: lift of the top 20 % 1.12 against 1.61 with P0, same network);
* target flip: BCE on the hard flip instead of the oracle's soft label (within-instance AUC on unseen
  instances 0.687 against 0.669, three seeds).

v3b = v3 + ``tabu``: single-edge moves already tried under the same search state go to the tail of the
next trial order. With the near-greedy P0 policy, v3 kept proposing the same top candidates on every
retry; the search skipped them as cached, so on instances where the model ranked the good moves low
the retries ran out with nothing new tried (PROTEINS ids 148, 454, 159 stayed at their initial GED).

Arms on the 40 evaluation instances:

    v3b            v3 with the tabu memory (online learning from the untrained network)
    v3             online learning from the untrained network
    v3-frozen      untrained seeded network, never updated (same policy)
    v3-pre         offline-pretrained network plus online learning
    v3-pre-frozen  offline-pretrained network, never updated

The uniform-proposal reference is v2e-random (P0 does not change a random policy).

plus ``v3-collect``: uniform proposals on the fold-0 test instances outside the evaluation set, dumping
every scored example for the offline benchmark and pretraining (``scripts/pretrain_selector_v3.py``).

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

V3 = {}   # the LBS skeleton stays exactly as in v2e
V3_SELECTOR = {"score_standardise": True, "temp_remove": 0.25, "target": "flip"}
ARMS = {
    "v3b": {"selector_mode": "learned", "selector": {"tabu": True}},
    "v3": {"selector_mode": "learned"},
    "v3-frozen": {"selector_mode": "frozen"},
    "v3-pre": {"selector_mode": "learned", "init_checkpoint": PRETRAINED},
    "v3-pre-frozen": {"selector_mode": "frozen", "init_checkpoint": PRETRAINED},
}


def make(arm: str, extra: dict, ids: list[int], scope: str) -> dict:
    conf = json.load(open(SRC))
    conf["experiment"]["scope"] = scope
    conf["evaluator"]["parameters"]["instance_ids"] = ids
    par = conf["doe-triplets"][0]["explainer"]["parameters"]["minimizer"]["parameters"]
    par.update(V3)
    extra = dict(extra)
    sel_extra = extra.pop("selector", {})
    par.update(extra)
    if arm != "v3-collect":
        par["selector"] = {**par.get("selector", {}), **V3_SELECTOR, **sel_extra}
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
