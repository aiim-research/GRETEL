#!/usr/bin/env python3
"""Clone selector-study configs with other minimizer seeds.

For each method directory and seed s it writes ``<method>-s<s>/generate_minimize0.jsonc`` next to the
original, with the minimizer ``seed`` set to s, the scope and ``model_tag`` suffixed with ``-s<s>`` (each
seed starts its own cold checkpoint) and, optionally, the shared generator cache. The generator stays
the same, so every seed of every method starts from the same counterfactual and the comparison stays
paired by instance; the seeds only resample the search and the selector.

    python scripts/gen_selector_seed_configs.py lab/config/generate_minimize/proteins40/dce \\
        --methods dce-lcls-net-v3b dce-lcls-net-v2e dce-lcls-net-v2e-random --seeds 1 2 \\
        --generator-cache lab/data/cache/generator_outputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root"); ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--generator-cache", default=None)
    args = ap.parse_args()
    for m in args.methods:
        base = json.load(open(Path(args.root) / m / "generate_minimize0.jsonc"))
        for sd in args.seeds:
            conf = json.loads(json.dumps(base))
            conf["experiment"]["scope"] += f"-s{sd}"
            expl = conf["doe-triplets"][0]["explainer"]["parameters"]
            if args.generator_cache:
                expl["generator_cache"] = args.generator_cache
            par = expl["minimizer"]["parameters"]
            par["seed"] = sd
            if "model_tag" in par:
                par["model_tag"] += f"-s{sd}"
            out = Path(args.root) / f"{m}-s{sd}" / "generate_minimize0.jsonc"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(conf, indent=4) + "\n")
            print(out, conf["experiment"]["scope"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
