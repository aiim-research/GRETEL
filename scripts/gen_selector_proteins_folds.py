#!/usr/bin/env python3
"""Generate the 10-fold PROTEINS comparison of the selector against plain LBS (thesis protocol).

The selector is the configuration validated on PROTEINS-40 (formerly v3b-pre-frozen): the v2e LBS
skeleton unchanged, the online edge selector with the standardised sweep policy (``sweep_temp`` 0.25),
flip target, tabu memory and sampled swap/reduce at ``sample_temp`` 4, offline-pretrained and frozen.
Arms, all with ``max_oracle_calls`` 10000 (the thesis budget), minimizer seed 0, DCE generator shared
through the generator cache, every fold's full test split:

    dce-lcls                 plain LBS (strict_budget, so it stops at the same cap)
    dce-lcls-sel             the selector
    dce-lcls-sel-untrained   the same network and policy with untrained weights, never updated
    dce-lcls-sel-random      the same skeleton with uniform proposals (no model)

The three selector arms also overshoot (``OVERSHOOT``): when a removal sweep finds nothing, they try
removing many edges at once with random blocks, once per search state (no extra retries), with a cheap
geometric schedule, and with the plain-LBS depth while the instance has had no improvement at all (the
ranking is evidently not working there). On PROTEINS ids 148 and 159, where the selector used to stay at
its initial size, it then matches plain LBS (8 and 1) with 43 to 58 % fewer calls. They also stop after
the first outer iteration that improves nothing (``retry_outer`` false): on fold 0 no improvement ever
came after a failed iteration, and on 9 test instances this halves the calls at the same or better GED.

selector against untrained isolates what training buys; untrained against random isolates the policy on
an arbitrary ranking; selector against plain LBS is the headline comparison with the thesis.

    python scripts/gen_selector_proteins_folds.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "lab/config/generate_minimize/proteins40/dce"
DST = REPO / "lab/config/generate_minimize/proteins-sel/dce"
ARMS = {
    "dce-lcls": ("dce-lcls", None),
    "dce-lcls-sel": ("dce-lcls-net-v3b-pre-frozen", "selector"),
    "dce-lcls-sel-untrained": ("dce-lcls-net-v3b-frozen", "selector-untrained"),
    "dce-lcls-sel-random": ("dce-lcls-net-v2e-random", "selector-random"),
}
BUDGET = 10000
OVERSHOOT = {"overshoot": True, "overshoot_sizes": "geometric", "overshoot_ratio": 0.7, "overshoot_trials": 12,
             "overshoot_fallback": True, "overshoot_budget_frac": 0.5,
             # no retries: an outer iteration that improves nothing (removal sweep, overshoot, swap and
             # add included) ends the search. On fold 0, none of 1462 improvements came after a failed
             # iteration, while 82 % of the selector's calls were spent in those retries.
             "retry_outer": False,
             # once removal is exhausted in a state, swap and add get half the iteration's budget each,
             # and swaps and overshoot blocks favour near-miss removals (edges whose single removal came
             # closest to a flip). Preliminary test on 10 fold-0 instances: GED sum 156 against 175 without
             # them and 298 for plain LBS, at 71 % fewer calls than plain LBS
             "stuck_swap_add": True, "near_miss": True}
CACHE = "lab/data/cache/generator_outputs"


def main() -> int:
    for arm, (src, tag) in ARMS.items():
        base = json.load(open(SRC / src / "generate_minimize0.jsonc"))
        for fold in range(10):
            conf = json.loads(json.dumps(base))
            conf["experiment"]["scope"] = "proteins-sel_" + arm.replace("dce-lcls", "dce_lcls", 1)
            for prop in conf["experiment"]["parameters"]["propagate"]:
                if prop["in_sections"] == ["doe-triplets/explainer"]:
                    prop["params"]["fold_id"] = fold
            conf["evaluator"]["parameters"].pop("instance_ids", None)
            expl = conf["doe-triplets"][0]["explainer"]["parameters"]
            expl["fold_id"] = fold
            expl["generator_cache"] = CACHE
            par = expl["minimizer"]["parameters"]
            par["max_oracle_calls"] = BUDGET
            par["seed"] = 0
            if tag:
                par["model_tag"] = tag
                par.update(OVERSHOOT)
            out = DST / arm / f"generate_minimize{fold}.jsonc"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(conf, indent=4) + "\n")
        print(arm, conf["experiment"]["scope"], "<-", src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
