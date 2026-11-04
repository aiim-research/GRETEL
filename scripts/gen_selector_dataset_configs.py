#!/usr/bin/env python3
"""Clone selector-study configs from the proteins40 tree onto another dataset.

Keeps every minimizer setting (budget, seeds, selector, arms) and swaps the dataset/oracle pair, the
scope prefix and the fixed evaluation instances, so the same methods run under the same protocol on a
new dataset. Pretrained arms take the dataset's own checkpoint (``--pretrained``). With
``--generator-cache`` every config stores and replays the generator's counterfactual there, so the
generator runs once per instance and all minimizers start from the same counterfactual.

    python scripts/gen_selector_dataset_configs.py --do lab/config/snippets/do-pairs/AIDS_GCN.json \\
        --prefix aids40 --instances lab/config/snippets/aids_40_instances.json \\
        --methods dce-lcls dce-lcls-net dce-lcls-net-v2e dce-lcls-net-v3 \\
        --generator-cache lab/data/cache/generator_outputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "lab/config/generate_minimize/proteins40/dce"
PROTEINS_PRE = "models/pretrained/edge_selector_v3-pre_PROTEINS.pt"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--do", required=True, help="dataset/oracle snippet")
    ap.add_argument("--prefix", required=True, help="scope and directory prefix, e.g. aids40")
    ap.add_argument("--instances", required=True)
    ap.add_argument("--methods", nargs="+", required=True, help="method directories under proteins40/dce")
    ap.add_argument("--pretrained", default=None, help="checkpoint for the *-pre arms of this dataset")
    ap.add_argument("--generator-cache", default=None, help="directory for the shared generator outputs")
    args = ap.parse_args()

    ids = [int(r["id"]) for r in json.load(open(args.instances))["instances"]]
    for m in args.methods:
        conf = json.load(open(SRC / m / "generate_minimize0.jsonc"))
        scope = conf["experiment"]["scope"]
        assert scope.startswith("proteins40_"), scope
        conf["experiment"]["scope"] = args.prefix + scope[len("proteins40"):]
        conf["doe-triplets"][0]["compose_do"] = "./" + str(Path(args.do))
        conf["evaluator"]["parameters"]["instance_ids"] = ids
        if args.generator_cache:
            conf["doe-triplets"][0]["explainer"]["parameters"]["generator_cache"] = args.generator_cache
        par = conf["doe-triplets"][0]["explainer"]["parameters"]["minimizer"]["parameters"]
        if par.get("init_checkpoint") == PROTEINS_PRE:
            assert args.pretrained, f"{m} needs --pretrained for this dataset"
            par["init_checkpoint"] = args.pretrained
        out = REPO / "lab/config/generate_minimize" / args.prefix / "dce" / m / "generate_minimize0.jsonc"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(conf, indent=4) + "\n")
        print(out.relative_to(REPO), conf["experiment"]["scope"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
