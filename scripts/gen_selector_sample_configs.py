#!/usr/bin/env python3
"""Derive a fixed-sample selector config tree from an existing one.

Copies every ``<src>/<method>/generate_minimize0.jsonc`` into ``<dst>/<method>/``, renaming the
scope prefix, pinning the evaluator to the instance ids of an ``instances`` snippet and setting the
minimizer budget. Plain LBS (``local_search.LocalSearch``) also gets ``strict_budget`` so that every
method stops at the same number of minimizer oracle calls. The thesis selector (v1) gets ``seed`` and
``monitor``. With ``--frozen-arms`` every learned selector also gets a ``-frozen`` twin that proposes
with the same untrained network and never updates it (learned against frozen = what training adds).

    python scripts/gen_selector_sample_configs.py \\
        lab/config/generate_minimize/proteins10/dce lab/config/generate_minimize/proteins40/dce \\
        --prefix proteins10 --new-prefix proteins40 \\
        --instances lab/config/snippets/proteins_40_instances.json --budget 2000 --frozen-arms
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_LBS = "src.explainer.future.metaheuristic.local_search.local_search.LocalSearch"
V1 = "src.explainer.future.metaheuristic.local_search.local_search_selection_net.LocalSearch"
V2 = "src.explainer.future.metaheuristic.local_search.local_search_selection_net_v2.LocalSearch"


def frozen_twin(conf: dict) -> dict | None:
    """The ``-frozen`` copy of a learned-selector config, or None if the config has no learned selector."""
    conf = json.loads(json.dumps(conf))
    mini = conf["doe-triplets"][0]["explainer"]["parameters"]["minimizer"]
    par = mini["parameters"]
    if mini["class"] == V1:
        par["freeze_selector"] = True
    elif mini["class"] == V2 and par.get("selector_mode", "learned") == "learned":
        par["selector_mode"] = "frozen"
        par["model_tag"] = par.get("model_tag", "v2") + "-frozen"
    else:
        return None
    conf["experiment"]["scope"] += "-frozen"
    return conf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--prefix", required=True); ap.add_argument("--new-prefix", required=True)
    ap.add_argument("--instances", required=True, help="snippet with an 'instances' list of {id, ...}")
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--frozen-arms", action="store_true")
    args = ap.parse_args()

    ids = [int(r["id"]) for r in json.load(open(args.instances))["instances"]]
    for cfg in sorted(Path(args.src).glob("*/generate_minimize0.jsonc")):
        conf = json.load(open(cfg))
        scope = conf["experiment"]["scope"]
        assert scope.startswith(args.prefix), scope
        conf["experiment"]["scope"] = args.new_prefix + scope[len(args.prefix):]
        conf["evaluator"]["parameters"]["instance_ids"] = ids
        mini = conf["doe-triplets"][0]["explainer"]["parameters"]["minimizer"]
        mini["parameters"]["max_oracle_calls"] = args.budget
        if mini["class"] == BASE_LBS:
            mini["parameters"]["strict_budget"] = True
        if mini["class"] == V1:
            mini["parameters"]["seed"] = 0
            mini["parameters"]["monitor"] = True
        variants = [(cfg.parent.name, conf)]
        twin = frozen_twin(conf) if args.frozen_arms else None
        if twin is not None:
            variants.append((cfg.parent.name + "-frozen", twin))
        for name, c in variants:
            out = Path(args.dst) / name / cfg.name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(c, indent=4) + "\n")
            print(out, c["experiment"]["scope"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
