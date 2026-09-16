#!/usr/bin/env python3
"""Batch experiment runner.

Runs every fold-config for a fixed set of (dataset × generator × minimizer)
combinations to completion (not a smoke test — the full test set is
processed). Results land under ``lab/output/results/<scope>/...`` exactly as
the configs declare; the script does not need to write them itself.

Each (dataset, combo, fold) runs in its own subprocess to dodge the
``Context.__global`` singleton trap. By default already-produced result
files are skipped so the script is restartable.

Usage from the repo root with the GRTL conda env active::

    python tests/run_experiments.py
    python tests/run_experiments.py --datasets bzr asd
    python tests/run_experiments.py --combos ofs/ofs-obs
    python tests/run_experiments.py --folds 0 1 2 --run-number 2
    python tests/run_experiments.py --force      # re-run even if results exist
    python tests/run_experiments.py --timeout 3600  # per-fold budget

Exit code is 0 if every config that ran finished cleanly.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DEFAULT_DATASETS = ["bzr", "asd", "enzymes", "synthie"]
DEFAULT_COMBOS = ["ofs/ofs-obs", "dfs/dfs-dbs"]
DEFAULT_FOLDS = list(range(10))

CFG_TMPL = "lab/config/generate_minimize/{ds}/{combo}/generate_minimize{fold}.jsonc"
RESULTS_ROOT = REPO / "lab" / "output" / "results"

# ---------------------------------------------------------------------------
# Child-process worker
# ---------------------------------------------------------------------------


def _child_main(rel_cfg: str, run_number: int) -> int:
    cfg_path = REPO / rel_cfg
    if not cfg_path.is_file():
        print(f"::RESULT:: status=MISSING_CFG", flush=True)
        return 2

    sys.path.insert(0, str(REPO))
    try:
        from src.utils.context import Context
        from src.evaluation.future.evaluator_manager_triplets import EvaluatorManager
    except Exception as e:
        print(f"::RESULT:: status=FAIL detail={type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
        return 3

    t0 = time.time()
    try:
        ctx = Context.get_context(str(cfg_path))
        ctx.run_number = run_number
        mgr = EvaluatorManager(ctx)
        mgr.evaluate()
        print(f"::RESULT:: status=OK detail=evaluate finished in {time.time()-t0:.1f}s",
              flush=True)
        return 0
    except Exception as e:
        print(f"::RESULT:: status=FAIL detail={type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
        return 4


# ---------------------------------------------------------------------------
# Helpers for the driver
# ---------------------------------------------------------------------------


def _scope_for(ds: str, combo: str) -> str:
    """Convert e.g. 'bzr', 'ofs/ofs-obs' → 'bzr_ofs_obs' (matches config scopes)."""
    gen, varname = combo.split("/", 1)
    if varname.startswith(gen + "-"):
        suffix = varname[len(gen) + 1:]
    else:
        suffix = varname
    return f"{ds}_{gen}_{suffix}"


def _results_already(ds: str, combo: str, fold: int, run_number: int) -> bool:
    """Check whether a results_{fold}_{run}.json already exists under the
    expected scope subtree. The dataset / oracle / explainer hash dirs are
    discovered at runtime, so we glob."""
    scope = _scope_for(ds, combo)
    scope_dir = RESULTS_ROOT / scope
    if not scope_dir.is_dir():
        return False
    target = f"results_{fold}_{run_number}.json"
    for f in scope_dir.rglob(target):
        return True
    return False


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def driver_main(args: argparse.Namespace) -> int:
    datasets = args.datasets or DEFAULT_DATASETS
    combos = args.combos or DEFAULT_COMBOS
    folds = args.folds if args.folds is not None else DEFAULT_FOLDS

    work: list[tuple[str, str, int]] = []
    for ds in datasets:
        for combo in combos:
            for fold in folds:
                work.append((ds, combo, fold))

    print(f"{len(work)} configs to run "
          f"(datasets={datasets}, combos={combos}, folds={folds}, "
          f"run_number={args.run_number}, timeout={args.timeout}s)\n",
          flush=True)

    n_ok = n_skip = n_bad = 0
    t_total = time.time()
    for i, (ds, combo, fold) in enumerate(work, start=1):
        tag = f"[{i:>3}/{len(work)}] {ds}/{combo}/fold={fold}"
        rel = CFG_TMPL.format(ds=ds, combo=combo, fold=fold)
        cfg_full = REPO / rel
        if not cfg_full.is_file():
            print(f"  · {tag:<55} MISSING_CFG ({rel})", flush=True)
            n_bad += 1
            continue
        if (not args.force) and _results_already(ds, combo, fold, args.run_number):
            print(f"  · {tag:<55} SKIP (results already exist)", flush=True)
            n_skip += 1
            continue

        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, __file__, "--one", rel,
                 "--run-number", str(args.run_number)],
                cwd=str(REPO),
                timeout=args.timeout,
                capture_output=True,
                text=True,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            line = next(
                (l for l in out.splitlines() if l.startswith("::RESULT::")),
                f"::RESULT:: status=NO_RESULT detail=exit_code={proc.returncode}",
            )
            status = line.split("status=", 1)[-1].split(" ", 1)[0]
            detail = line.split("detail=", 1)[-1] if "detail=" in line else ""
            elapsed = time.time() - t0
            mark = "✓" if status == "OK" else "✗"
            print(f"  {mark} {tag:<55} {status:<6} {detail[:50]:<55} [{elapsed:5.0f}s]",
                  flush=True)
            if status == "OK":
                n_ok += 1
            else:
                n_bad += 1
                if args.verbose:
                    print("    ---- subprocess tail ----")
                    for ln in out.splitlines()[-25:]:
                        print(f"    {ln}")
                    print("    ---- end ----")
                if args.fail_fast:
                    break
        except subprocess.TimeoutExpired:
            n_bad += 1
            print(f"  ✗ {tag:<55} TIMEOUT after {args.timeout}s", flush=True)
            if args.fail_fast:
                break

    print(f"\n== Summary ==  ok={n_ok}  skipped={n_skip}  failed={n_bad}  "
          f"total={len(work)}  wallclock={time.time()-t_total:.0f}s",
          flush=True)
    return 0 if n_bad == 0 else 1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--one", default=None,
                    help="(child mode) run a single config to completion")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help=f"default: {DEFAULT_DATASETS}")
    ap.add_argument("--combos", nargs="+", default=None,
                    help=f"default: {DEFAULT_COMBOS}")
    ap.add_argument("--folds", nargs="+", type=int, default=None,
                    help="default: 0..9")
    ap.add_argument("--run-number", type=int, default=1,
                    help="written into results_<fold>_<run>.json (default 1)")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="per-fold wall-clock budget in seconds (default 3600)")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if a result file already exists")
    ap.add_argument("--fail-fast", action="store_true",
                    help="stop at the first failed/timed-out combo")
    ap.add_argument("--verbose", action="store_true",
                    help="dump subprocess tail on failure")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if a.one:
        sys.exit(_child_main(a.one, a.run_number))
    sys.exit(driver_main(a))
