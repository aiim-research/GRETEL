#!/usr/bin/env python3
"""List the configs of one or more queue files that still have no results, one path per line.

A queue file is the markdown task list used by ``scripts/run_revision_queue.py`` (``- [ ] <config>``).
A config counts as done when ``lab/output/results/<scope>/**/results_<fold>_<run>.json`` exists, whatever
its checkbox says, so the list is always computed against what is actually on disk. Used by
``launchers/slurm_submit.sh`` to build the Slurm job array.

    python scripts/pending_configs.py docs/selector/SERVER_PROTEINS_FOLDS.md > /tmp/pending.list
    python scripts/pending_configs.py --where <config>   # "<scope> <fold> <log folder>", for slurm_job.sh
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from result_paths import combo_dir  # noqa: E402
from run_revision_queue import _fold_of, _parse_list, _results_already, _scope_of  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("queues", nargs="*", help="queue markdown files")
    ap.add_argument("--run-number", type=int, default=1)
    ap.add_argument("--where", metavar="CONFIG", help="print the scope, fold and log folder of one config")
    args = ap.parse_args()
    if args.where:
        scope = _scope_of(args.where)
        print(scope, _fold_of(args.where), combo_dir(scope))
        return 0
    pending, done = [], 0
    for q in args.queues:
        _, tasks = _parse_list(Path(q))
        for t in tasks:
            cfg = t["path"]
            if _results_already(_scope_of(cfg), _fold_of(cfg), args.run_number):
                done += 1
            elif cfg not in pending:
                pending.append(cfg)
    for cfg in pending:
        print(cfg)
    print(f"{len(pending)} pending, {done} already done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
