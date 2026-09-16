#!/usr/bin/env python3
"""Standalone DCM trainer.

Computes a DCM (Distance-based Class Medoid) model for a given dataset + oracle
combination, *outside* the generate_minimize pipeline. One medoid per class,
saved as a tiny pickle keyed by the dataset hash.

The script accepts either:
  * --do-pair  PATH    a do-pair snippet under lab/config/snippets/do-pairs/
                       (the dataset + oracle definition, e.g. ASD_ASD-Custom.json)
  * --config   PATH    any full generate-minimize config (the script extracts the
                       first triplet's dataset + oracle from it)

It then builds a minimal Context, instantiates the dataset and oracle the same
way the framework does, constructs a DCM explainer with the requested
``proportion`` (0..1) of the dataset used as medoid candidates, and triggers
its fit/save. The saved file path is the same one any DCM-using config later
loads from, so a subsequent generate_minimize run picks it up automatically.

Usage from the repo root with the GRTL-cpu (or GRTL) env active::

    # full dataset
    python scripts/compute_dcm.py --do-pair lab/config/snippets/do-pairs/BZR_GCN.json \
                                  --proportion 1.0

    # half the dataset, classify candidates by oracle prediction
    python scripts/compute_dcm.py --config lab/config/generate_minimize/asd/dcm/dcm-lcls/generate_minimize0.jsonc \
                                  --proportion 0.5 --classify-with oracle

    # quick smoke train (a tiny sample)
    python scripts/compute_dcm.py --do-pair ... --proportion 0.05 --force

Exit code 0 on success, non-zero otherwise.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_jsonc(path: Path) -> dict:
    """Lightweight JSONC reader (strips // line comments and /* */ blocks)."""
    txt = path.read_text()
    txt = re.sub(r"/\*.*?\*/", "", txt, flags=re.DOTALL)
    txt = re.sub(r"//.*?\n", "\n", txt)
    return json.loads(txt)


def _build_synthetic_config(
    dataset_snippet: dict,
    oracle_snippet: dict,
    proportion: float,
    classify_with: str,
    force: bool,
) -> dict:
    """Wrap a do-pair into a minimal generate_minimize-style config so the
    framework's Context loader is happy. The explainer block is a DCM with
    the requested params; the evaluator is omitted because we only need fit()."""
    return {
        "experiment": {
            "scope": "dcm_train",
            "parameters": {
                "lock_release_tout": 120,
                "propagate": [
                    {"in_sections": ["doe-triplets/oracle"],
                     "params": {"fold_id": -1, "retrain": False}},
                ],
            },
        },
        "doe-triplets": [
            {
                "dataset": dataset_snippet,
                "oracle": oracle_snippet,
                "explainer": {
                    "class": "src.explainer.future.search.dcm.DCM",
                    "parameters": {
                        "fold_id": -1,
                        "proportion": float(proportion),
                        "classify_with": classify_with,
                        "retrain": bool(force),
                    },
                },
            }
        ],
        "compose_strs": "./lab/config/snippets/default_store_paths.json",
    }


def _resolve_compose(snippet_path: Path) -> dict:
    """Resolve a do-pair file. If it uses compose_do to point at another file,
    follow it. Otherwise return its contents directly."""
    cfg = _read_jsonc(snippet_path)
    if "dataset" in cfg and "oracle" in cfg:
        return cfg
    if "compose_do" in cfg:
        target = REPO / cfg["compose_do"].lstrip("./")
        return _resolve_compose(target)
    raise ValueError(f"{snippet_path} does not look like a do-pair (no dataset/oracle keys)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--do-pair",
                     help="path to a do-pair snippet (e.g. lab/config/snippets/do-pairs/BZR_GCN.json)")
    src.add_argument("--config",
                     help="path to any generate_minimize config; its first triplet is used")
    ap.add_argument("--proportion", type=float, default=1.0,
                    help="fraction of each class used as medoid candidates: "
                         "0 = nothing, 1 = full (default 1.0)")
    ap.add_argument("--classify-with", choices=["label", "oracle"], default="label",
                    help="how to assign each graph to a class during training "
                         "('label' is fast; 'oracle' follows the thesis but adds N "
                         "oracle calls during the offline phase)")
    ap.add_argument("--force", action="store_true",
                    help="retrain even if a saved DCM with the same hash exists")
    ap.add_argument("--manipulator",
                    default="lab/config/snippets/datasets/empty.json",
                    help="path to a dataset manipulator snippet (default: empty.json)")
    args = ap.parse_args()

    if not 0 <= args.proportion <= 1:
        ap.error("--proportion must be in [0, 1]")

    # Pull dataset + oracle definitions out of either source.
    if args.do_pair:
        do_pair = _resolve_compose(REPO / args.do_pair)
    else:
        cfg = _read_jsonc(REPO / args.config)
        triplet = cfg["doe-triplets"][0]
        if "compose_do" in triplet:
            do_pair = _resolve_compose(REPO / triplet["compose_do"].lstrip("./"))
        else:
            do_pair = {"dataset": triplet["dataset"], "oracle": triplet["oracle"]}

    synth = _build_synthetic_config(
        dataset_snippet=do_pair["dataset"],
        oracle_snippet=do_pair["oracle"],
        proportion=args.proportion,
        classify_with=args.classify_with,
        force=args.force,
    )

    # Attach the manipulator (padding etc.) the same way generate_minimize configs do.
    synth["experiment"]["parameters"]["propagate"].append(
        {"in_sections": ["doe-triplets/dataset"],
         "params": {"compose_man": args.manipulator}}
    )

    # Drop the synthetic config to a tmp .jsonc and feed it to Context.
    with tempfile.NamedTemporaryFile("w", suffix=".jsonc", delete=False) as f:
        json.dump(synth, f, indent=2)
        tmp_path = Path(f.name)
    print(f"Built synthetic DCM config: {tmp_path}")

    try:
        from src.utils.context import Context
        from src.dataset.dataset_factory import DatasetFactory
        from src.oracle.oracle_factory import OracleFactory
        from src.explainer.explainer_factory import ExplainerFactory
        from src.oracle.embedder_factory import EmbedderFactory

        ctx = Context.get_context(str(tmp_path))
        ctx.run_number = 0
        ctx.factories["datasets"] = DatasetFactory(ctx)
        ctx.factories["embedders"] = EmbedderFactory(ctx)
        ctx.factories["oracles"] = OracleFactory(ctx)
        ctx.factories["explainers"] = ExplainerFactory(ctx)

        triplet = ctx.conf["doe-triplets"][0]
        print("Loading dataset …")
        dataset = ctx.factories["datasets"].get_dataset(triplet["dataset"])
        print(f"  -> {dataset.name}  ({len(dataset.instances)} instances)")
        print("Loading/Building oracle (retrain=False by default) …")
        oracle = ctx.factories["oracles"].get_oracle(triplet["oracle"], dataset)
        print(f"  -> {oracle.name}")
        print(f"Training DCM (proportion={args.proportion}, "
              f"classify_with={args.classify_with}, force={args.force}) …")
        explainer = ctx.factories["explainers"].get_explainer(
            triplet["explainer"], dataset, oracle
        )

        # The factory triggers Trainable.load_or_create() in __init__. If a
        # cached DCM exists with the same hash, training is skipped automatically;
        # with --force we passed retrain=True so it always retrains.
        save_path = ctx.get_path(explainer)
        print(f"DCM ready at: {save_path}")
        print(f"Medoids per class: {explainer.model}")
        return 0
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
