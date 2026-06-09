#!/usr/bin/env python3
"""Generate config files for the REVISION_EXPERIMENTS.md plan.

Creates two batches under ``lab/config/generate_minimize/`` by cloning
already-existing templates and injecting the needed parameter changes:

* **Exp. 1 (multi-semilla)**: clone every ``<ds>/<gen>/<gen>-<min>/``
  combo (min in {lcls, obs}) into ``<ds>/<gen>/<gen>-<min>-seed<s>/`` for
  s in {0, 1, 2, 3, 4}, injecting ``seed: <s>`` into the minimizer params
  and rewriting the experiment scope to ``<ds>_<gen>_<min>_seed<s>``.

* **Exp. 3b (random hill-climbing)**: clone every ``<ds>/<gen>/<gen>-lcls/``
  template into ``<ds>/<gen>/<gen>-rhc/``, swapping the minimizer class to
  ``RandomHillClimbing`` (sibling of LocalSearch) with matching budget and
  ``manip_attr: false`` so attribute manipulation is left to the LBS run.

Run from the repo root::

    python scripts/gen_revision_configs.py
    python scripts/gen_revision_configs.py --dry-run     # print plan, write nothing
    python scripts/gen_revision_configs.py --exp 1       # only Exp. 1
    python scripts/gen_revision_configs.py --exp 3b      # only Exp. 3b

Idempotent: re-running overwrites the generated files (the source
templates are never modified).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CFG_ROOT = REPO / "lab" / "config" / "generate_minimize"

DATASETS = ["synthie", "asd", "bbbp", "enzymes", "bzr", "aids", "tcr-tco-300"]
GENERATORS = ["dce", "ofs", "rsgg"]
MINIMIZERS = ["lcls", "obs"]
SEEDS = [0, 1, 2, 3, 4]
FOLDS = list(range(10))

RHC_CLASS = "src.explainer.future.metaheuristic.local_search.random_hill_climbing.RandomHillClimbing"


def _read_jsonc(path: Path) -> "OrderedDict":
    """Read a JSONC file (strips ``//`` and ``/* ... */`` comments) preserving key order."""
    txt = path.read_text()
    txt = re.sub(r"/\*.*?\*/", "", txt, flags=re.DOTALL)
    txt = re.sub(r"//.*?\n", "\n", txt)
    return json.loads(txt, object_pairs_hook=OrderedDict)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)


def _patch_seed(cfg, scope: str, seed: int):
    """Set the experiment scope and inject ``seed`` into the minimizer params."""
    cfg["experiment"]["scope"] = scope
    for triplet in cfg["doe-triplets"]:
        minim = triplet["explainer"]["parameters"]["minimizer"]
        minim["parameters"]["seed"] = int(seed)


def _swap_to_rhc(cfg, scope: str, *, patience: int = 40):
    """Convert an LBS template into a RandomHillClimbing template.

    Keeps the LBS oracle-call budget and ``attributed`` flag so the RHC
    head-to-head against LBS is at equal budget. The four LBS-only knobs
    (neigh_factor, runtime_factor, max_runtime, max_neigh) are dropped
    because RHC samples one neighbor per step rather than sweeping.
    """
    cfg["experiment"]["scope"] = scope
    for triplet in cfg["doe-triplets"]:
        minim = triplet["explainer"]["parameters"]["minimizer"]
        lbs_params = minim.get("parameters", {})
        attributed = bool(lbs_params.get("attributed", False))
        # The LBS template doesn't always carry max_oracle_calls (it
        # defaults to 10000 in code). Mirror that default explicitly here
        # so the RHC budget is the same on disk as LBS uses at runtime.
        max_oc = int(lbs_params.get("max_oracle_calls", 10000))

        minim["class"] = RHC_CLASS
        minim["parameters"] = OrderedDict([
            ("attributed", attributed),
            ("manip_attr", False),
            ("max_oracle_calls", max_oc),
            ("patience", patience),
        ])


def plan_exp1():
    """Yield (template, target, scope, seed) for every Exp. 1 file."""
    for ds in DATASETS:
        for gen in GENERATORS:
            for mn in MINIMIZERS:
                src_dir = CFG_ROOT / ds / gen / f"{gen}-{mn}"
                if not src_dir.is_dir():
                    print(f"  skip (template missing): {src_dir}", file=sys.stderr)
                    continue
                for seed in SEEDS:
                    dst_dir = CFG_ROOT / ds / gen / f"{gen}-{mn}-seed{seed}"
                    scope = f"{ds}_{gen}_{mn}_seed{seed}"
                    for fold in FOLDS:
                        src = src_dir / f"generate_minimize{fold}.jsonc"
                        dst = dst_dir / f"generate_minimize{fold}.jsonc"
                        if not src.is_file():
                            print(f"  skip (fold missing): {src}", file=sys.stderr)
                            continue
                        yield src, dst, scope, seed


def plan_exp3b():
    """Yield (template, target, scope) for every Exp. 3b file."""
    for ds in DATASETS:
        for gen in GENERATORS:
            src_dir = CFG_ROOT / ds / gen / f"{gen}-lcls"
            if not src_dir.is_dir():
                print(f"  skip (template missing): {src_dir}", file=sys.stderr)
                continue
            dst_dir = CFG_ROOT / ds / gen / f"{gen}-rhc"
            scope = f"{ds}_{gen}_rhc"
            for fold in FOLDS:
                src = src_dir / f"generate_minimize{fold}.jsonc"
                dst = dst_dir / f"generate_minimize{fold}.jsonc"
                if not src.is_file():
                    print(f"  skip (fold missing): {src}", file=sys.stderr)
                    continue
                yield src, dst, scope


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp", choices=["1", "3b", "all"], default="all")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, write nothing")
    args = ap.parse_args()

    do_1 = args.exp in ("1", "all")
    do_3b = args.exp in ("3b", "all")
    n_written_1 = n_written_3b = 0

    if do_1:
        print("== Exp. 1 (multi-semilla) ==")
        for src, dst, scope, seed in plan_exp1():
            if args.dry_run:
                print(f"  {dst.relative_to(REPO)}  <-  {src.name}  (scope={scope}, seed={seed})")
                continue
            cfg = _read_jsonc(src)
            _patch_seed(cfg, scope, seed)
            _write_json(dst, cfg)
            n_written_1 += 1
        print(f"  -> wrote {n_written_1} files")

    if do_3b:
        print("== Exp. 3b (random hill-climbing) ==")
        for src, dst, scope in plan_exp3b():
            if args.dry_run:
                print(f"  {dst.relative_to(REPO)}  <-  {src.name}  (scope={scope})")
                continue
            cfg = _read_jsonc(src)
            _swap_to_rhc(cfg, scope)
            _write_json(dst, cfg)
            n_written_3b += 1
        print(f"  -> wrote {n_written_3b} files")

    print(f"\nDone. exp1={n_written_1}, exp3b={n_written_3b}, "
          f"total={n_written_1 + n_written_3b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
