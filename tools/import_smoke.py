#!/usr/bin/env python3
"""Import every module under ``src/`` and report the ones that fail.

A cheap guard against a reorganisation leaving a dangling ``from src.x import
y``: a rename that misses one reference shows up here immediately, without
running an experiment.

Some modules legitimately fail on a given machine because an optional heavy
dependency is missing (CUDA-only builds, ``dgl``, ``exmol``, ...).  Those are
reported separately as SKIPPED so a real breakage stays visible.

Usage::

    python tools/import_smoke.py            # every module
    python tools/import_smoke.py src/explainer   # only this subtree
"""
from __future__ import annotations

import importlib
import os
import sys
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# Import errors caused by an absent third-party package, not by our layout.
OPTIONAL = tuple(f"No module named '{p}'" for p in (
    "dgl", "exmol", "selfies", "karateclub", "rdkit", "torch_geometric",
    "google", "transformers", "picologging", "flufl", "spacy",
    "huggingface_hub", "hnswlib", "omegaconf", "gensim",
))

BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "import_smoke_baseline.txt")


def modules(subtree):
    root = os.path.join(REPO, subtree)
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in sorted(fn):
            if f.endswith(".py"):
                rel = os.path.relpath(os.path.join(dp, f), REPO)
                yield rel[:-3].replace(os.sep, ".")


def main():
    subtree = sys.argv[1] if len(sys.argv) > 1 else "src"
    failed, skipped, ok = [], [], 0
    for mod in modules(subtree):
        if mod.endswith(".__init__"):
            mod = mod[: -len(".__init__")]
        try:
            importlib.import_module(mod)
            ok += 1
        except BaseException as exc:  # noqa: BLE001 - we want every failure
            msg = f"{type(exc).__name__}: {exc}"
            if any(o in msg for o in OPTIONAL):
                skipped.append((mod, msg))
            else:
                failed.append((mod, msg, traceback.format_exc()))

    known = set()
    if os.path.isfile(BASELINE):
        known = {ln.split("#", 1)[0].strip() for ln in open(BASELINE)}
        known.discard("")
    new = [f for f in failed if f[0] not in known]

    print(f"imported OK   : {ok}")
    print(f"skipped       : {len(skipped)}  (optional dependency missing)")
    print(f"failed (known): {len(failed) - len(new)}")
    print(f"FAILED (new)  : {len(new)}")
    for mod, msg, tb in new:
        print(f"\n--- {mod}\n{tb}")
    if not new:
        print("\nOK: no module broke that was not already broken.")
    return 1 if new else 0


if __name__ == "__main__":
    raise SystemExit(main())
