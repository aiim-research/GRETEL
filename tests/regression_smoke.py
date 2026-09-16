#!/usr/bin/env python3
"""Regression smoke-test for GRETEL configurations.

For every (dataset × generator × non-trainable minimizer) combination listed
in MATRIX, this script loads the corresponding config under
``lab/config/generate_minimize/`` and runs the pipeline for exactly **one**
graph instance, then bails out. A failure surfaces three categories:

  * FAIL       — an exception from the framework or a class under test
  * TIMEOUT    — exceeded the per-combo wall-clock budget (default 120s)
  * MISSING    — the config tree expected by the matrix doesn't exist

Why subprocesses
----------------
``Context.get_context(cfg)`` stores the first Context on the class via
``Context.__global`` and returns it for every later call regardless of the
``cfg`` argument. Looping over configs in a single process would silently
reuse whichever dataset was loaded first. The driver therefore spawns one
subprocess per (dataset, gen, variant) combination.

Usage
-----
From the repo root with the GRTL conda env active::

    python tests/regression_smoke.py                       # full matrix
    python tests/regression_smoke.py --datasets asd bzr    # subset
    python tests/regression_smoke.py --gens dfs --vars dbs # smoke one combo
    python tests/regression_smoke.py --timeout 60          # tighter budget
    python tests/regression_smoke.py --fail-fast           # stop on first ✗

Exit codes
----------
  0 — all combinations passed (every config that exists)
  1 — at least one combination failed

The script is configuration-driven; when you add a new dataset, generator,
or variant to the layout, append it to the constants below.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable

# Repo root: tests/ lives one level under it.
REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Matrix definition (edit when the config tree changes)
# ---------------------------------------------------------------------------

DATASETS = [
    "aids", "asd", "bbbp", "bbbp-no-attr", "bzr", "colors-3", "cuneiform",
    "enzymes", "imdb", "proteins", "synthie", "synthie-no-att",
    "tcr-gcn", "tcr-tco-300",
]

GENERATORS = ["dce", "dcm", "ofs", "rsgg", "dfs"]

# Non-trainable variants only — trainable/ponderation variants need an
# internal model trained on first use, which exceeds a smoke-test budget.
VARIANTS = ["dummy", "lcls", "lcls-net", "lcls-var-1", "lcls-var-2", "obs", "dbs"]

CFG_TMPL = "lab/config/generate_minimize/{ds}/{gen}/{gen}-{variant}/generate_minimize0.jsonc"

# ---------------------------------------------------------------------------
# Child-process worker
# ---------------------------------------------------------------------------


class _StopAfterOne(Exception):
    """Sentinel raised after the first instance is evaluated to bail out."""


def _child_main(rel_cfg: str) -> int:
    cfg_path = REPO / rel_cfg
    if not cfg_path.is_file():
        print(f"::RESULT:: status=MISSING")
        return 2

    # Imports happen inside the child so a broken import shows up per-combo.
    try:
        sys.path.insert(0, str(REPO))
        from src.utils.context import Context
        from src.evaluation.future.evaluator_manager_triplets import EvaluatorManager
        from src.evaluation.future.evaluator import Evaluator
    except Exception as e:
        print(f"::RESULT:: status=FAIL detail={type(e).__name__}: {e}")
        traceback.print_exc()
        return 3

    orig = Evaluator._real_evaluate

    def patched(self, instance):
        try:
            orig(self, instance)
        finally:
            raise _StopAfterOne(str(getattr(instance, "id", "?")))

    Evaluator._real_evaluate = patched

    t0 = time.time()
    try:
        ctx = Context.get_context(str(cfg_path))
        ctx.run_number = 0
        mgr = EvaluatorManager(ctx)
        try:
            mgr.evaluate()
            # An empty test set should still be reported as an error so the
            # config doesn't silently no-op.
            print("::RESULT:: status=FAIL detail=evaluate returned without instances")
            return 1
        except _StopAfterOne as s:
            print(f"::RESULT:: status=OK detail=id={s} in {time.time()-t0:.1f}s")
            return 0
    except Exception as e:
        print(f"::RESULT:: status=FAIL detail={type(e).__name__}: {e}")
        traceback.print_exc()
        return 4


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _filter(items: list[str], wanted: Iterable[str] | None) -> list[str]:
    if not wanted:
        return items
    wset = set(wanted)
    return [x for x in items if x in wset]


def _enumerate(datasets, generators, variants):
    for ds in datasets:
        for gen in generators:
            for var in variants:
                yield ds, gen, var


def _print_row(combo: str, status: str, detail: str = ""):
    mark = {"OK": "✓", "MISSING": "·", "FAIL": "✗", "TIMEOUT": "✗"}.get(status, "?")
    print(f"  {mark} {combo:<35} {status:<10} {detail[:90]}", flush=True)


def driver_main(args: argparse.Namespace) -> int:
    datasets = _filter(DATASETS, args.datasets)
    gens = _filter(GENERATORS, args.gens)
    vars_ = _filter(VARIANTS, args.vars)

    rows: list[tuple[str, str, str]] = []
    n_total = sum(1 for _ in _enumerate(datasets, gens, vars_))
    n_done = 0

    print(f"Running {n_total} (dataset × generator × variant) smoke tests, "
          f"{args.timeout}s budget each.\n", flush=True)

    for ds, gen, var in _enumerate(datasets, gens, vars_):
        n_done += 1
        rel = CFG_TMPL.format(ds=ds, gen=gen, variant=var)
        combo = f"[{n_done:>3}/{n_total}] {ds}/{gen}/{gen}-{var}"
        cfg_path = REPO / rel
        if not cfg_path.is_file():
            rows.append((combo, "MISSING", ""))
            _print_row(combo, "MISSING")
            continue

        try:
            proc = subprocess.run(
                [sys.executable, __file__, "--one", rel],
                cwd=str(REPO),
                timeout=args.timeout,
                capture_output=True,
                text=True,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            line = next(
                (l for l in out.splitlines() if l.startswith("::RESULT::")),
                f"::RESULT:: status=NO_RESULT_LINE detail=exit_code={proc.returncode}",
            )
            status = line.split("status=", 1)[-1].split(" ", 1)[0]
            detail = line.split("detail=", 1)[-1] if "detail=" in line else ""
            rows.append((combo, status, detail))
            _print_row(combo, status, detail)
            if status not in ("OK", "MISSING") and args.verbose:
                print("    ---- subprocess tail ----")
                for ln in out.splitlines()[-20:]:
                    print(f"    {ln}")
                print("    ---- end ----")
            if status not in ("OK", "MISSING") and args.fail_fast:
                break
        except subprocess.TimeoutExpired:
            rows.append((combo, "TIMEOUT", f">{args.timeout}s"))
            _print_row(combo, "TIMEOUT", f">{args.timeout}s")
            if args.fail_fast:
                break

    print("\n========== SUMMARY ==========")
    n_ok = sum(1 for _, s, _ in rows if s == "OK")
    n_miss = sum(1 for _, s, _ in rows if s == "MISSING")
    n_bad = len(rows) - n_ok - n_miss
    for combo, status, detail in rows:
        _print_row(combo, status, detail)
    print(f"\nPassed: {n_ok}   Missing: {n_miss}   Failed: {n_bad}   Total: {len(rows)}")
    return 0 if n_bad == 0 else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--one", default=None,
                    help="(child mode) run a single config and print ::RESULT::")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="restrict to these dataset names")
    ap.add_argument("--gens", nargs="+", default=None,
                    help="restrict to these generator names")
    ap.add_argument("--vars", nargs="+", default=None,
                    help="restrict to these variant suffixes")
    ap.add_argument("--timeout", type=int, default=120,
                    help="per-combo wall-clock budget in seconds (default 120)")
    ap.add_argument("--fail-fast", action="store_true",
                    help="stop at the first failed/timed-out combo")
    ap.add_argument("--verbose", action="store_true",
                    help="dump the subprocess tail on failure")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if a.one:
        sys.exit(_child_main(a.one))
    sys.exit(driver_main(a))
