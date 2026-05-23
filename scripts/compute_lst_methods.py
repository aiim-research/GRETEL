#!/usr/bin/env python3
"""Standalone trainer for the shared LST node-attribute-method artifact.

Trains :class:`LSTMethodsArtifact` for a given dataset + oracle and saves a
tiny pickle keyed by the dataset hash. Every trainable LocalSearch variant
(``local_search_trainable``, ``local_search_trainable_ponderation``,
``local_search_selection_net_trainable``) loads this artifact at init time,
so this script only needs to run once per dataset and the result is shared.

Use the sibling :mod:`scripts/compute_dcm.py` to train the medoids artifact;
the two together are everything a trainable LST variant needs.

Usage (from the repo root with GRTL-cpu active)::

    python scripts/compute_lst_methods.py --do-pair lab/config/snippets/do-pairs/BZR_GCN.json --proportion 1.0
    python scripts/compute_lst_methods.py --do-pair ASD_ASD-Custom.json --proportion 0.5 --force
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(REPO))


def _read_jsonc(path: Path) -> dict:
    txt = path.read_text()
    txt = re.sub(r"/\*.*?\*/", "", txt, flags=re.DOTALL)
    txt = re.sub(r"//.*?\n", "\n", txt)
    return json.loads(txt)


def _resolve_compose(snippet_path: Path) -> dict:
    cfg = _read_jsonc(snippet_path)
    if "dataset" in cfg and "oracle" in cfg:
        return cfg
    if "compose_do" in cfg:
        target = REPO / cfg["compose_do"].lstrip("./")
        return _resolve_compose(target)
    raise ValueError(f"{snippet_path} is not a do-pair (no dataset/oracle keys)")


def _synthetic_config(do_pair: dict, proportion: float, force: bool, manipulator: str) -> dict:
    return {
        "experiment": {
            "scope": "lst_methods_train",
            "parameters": {
                "lock_release_tout": 120,
                "propagate": [
                    {"in_sections": ["doe-triplets/oracle"],
                     "params": {"fold_id": -1, "retrain": False}},
                    {"in_sections": ["doe-triplets/dataset"],
                     "params": {"compose_man": manipulator}},
                ],
            },
        },
        "doe-triplets": [
            {
                "dataset": do_pair["dataset"],
                "oracle": do_pair["oracle"],
                "explainer": {
                    "class": "src.explainer.future.metaheuristic.local_search.lst_shared.LSTMethodsArtifact",
                    "parameters": {
                        "fold_id": -1,
                        "proportion": float(proportion),
                        "retrain": bool(force),
                    },
                },
            }
        ],
        "compose_strs": "./lab/config/snippets/default_store_paths.json",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--do-pair",
                     help="path to a do-pair snippet (e.g. lab/config/snippets/do-pairs/BZR_GCN.json)")
    src.add_argument("--config",
                     help="path to any generate_minimize config; its first triplet is used")
    ap.add_argument("--proportion", type=float, default=1.0,
                    help="fraction of dataset instances to score methods over (default 1.0)")
    ap.add_argument("--force", action="store_true",
                    help="retrain even if a cached artifact exists")
    ap.add_argument("--manipulator",
                    default="lab/config/snippets/datasets/padding.json",
                    help="dataset manipulator snippet (default: padding.json)")
    args = ap.parse_args()

    if not 0 <= args.proportion <= 1:
        ap.error("--proportion must be in [0, 1]")

    if args.do_pair:
        do_pair = _resolve_compose(REPO / args.do_pair)
    else:
        cfg = _read_jsonc(REPO / args.config)
        triplet = cfg["doe-triplets"][0]
        if "compose_do" in triplet:
            do_pair = _resolve_compose(REPO / triplet["compose_do"].lstrip("./"))
        else:
            do_pair = {"dataset": triplet["dataset"], "oracle": triplet["oracle"]}

    synth = _synthetic_config(do_pair, args.proportion, args.force, args.manipulator)

    with tempfile.NamedTemporaryFile("w", suffix=".jsonc", delete=False) as f:
        json.dump(synth, f, indent=2)
        tmp_path = Path(f.name)
    print(f"Built synthetic LST-methods config: {tmp_path}")

    try:
        from src.utils.context import Context
        from src.dataset.dataset_factory import DatasetFactory
        from src.oracle.oracle_factory import OracleFactory
        from src.oracle.embedder_factory import EmbedderFactory
        from src.explainer.future.metaheuristic.local_search.lst_shared import (
            LSTMethodsArtifact,
        )

        ctx = Context.get_context(str(tmp_path))
        ctx.run_number = 0
        ctx.factories["datasets"] = DatasetFactory(ctx)
        ctx.factories["embedders"] = EmbedderFactory(ctx)
        ctx.factories["oracles"] = OracleFactory(ctx)

        triplet = ctx.conf["doe-triplets"][0]
        print("Loading dataset …")
        dataset = ctx.factories["datasets"].get_dataset(triplet["dataset"])
        print(f"  -> {dataset.name}  ({len(dataset.instances)} instances)")
        print("Loading oracle …")
        oracle = ctx.factories["oracles"].get_oracle(triplet["oracle"], dataset)
        print(f"  -> {oracle.name}")

        print(f"Training methods artifact (proportion={args.proportion}, force={args.force}) …")
        artifact = LSTMethodsArtifact(
            context=ctx,
            local_config={
                "class": triplet["explainer"]["class"],
                "dataset": dataset,
                "oracle": oracle,
                "parameters": dict(triplet["explainer"]["parameters"]),
            },
        )
        save_path = ctx.get_path(artifact)
        print(f"Methods artifact ready at: {save_path}")
        print(f"Methods: {artifact.model['methods']}")
        return 0
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
