#!/usr/bin/env python3
"""Put the run logs of finished configs, gzipped, where git tracks them and the analysis scripts read them.

Tracked run logs live in ``lab/output/queue_logs/<dataset>/<generator>-<minimizer>/fold<f>.log.gz``
(``result_paths.run_log``). Two sources:

* Slurm job logs brought back from the server (``output/logs/<dataset>/<generator>-<minimizer>/
  fold<f>_<job>_<task>.out``, written by ``launchers/slurm_job.sh``; each starts with ``# config <path>``
  and ends with ``# finished ... with exit status N``). When a config ran more than once, the last attempt
  that finished with status 0 wins; configs with no successful attempt are listed and skipped.
* ``--local``: the plain logs the local queue runner leaves in ``lab/output/queue_logs``. Each one whose
  run has its results file is gzipped in place; the others (running, failed) are left alone.

    rsync -av <user>@<server>:<path>/GRETEL/output/logs/ lab/output/server_logs/
    python scripts/collect_logs.py lab/output/server_logs
    python scripts/collect_logs.py --local
"""
from __future__ import annotations

import argparse
import gzip
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from result_paths import QUEUE_LOGS, REPO, run_log  # noqa: E402
from run_revision_queue import _fold_of, _results_already, _scope_of  # noqa: E402

CONFIG = re.compile(r"^# config:? (\S+)")
FINISHED = re.compile(r"^# finished .* with exit status (\d+)")


def _header(path: Path) -> tuple[str | None, int | None]:
    cfg, status = None, None
    with open(path, errors="replace") as f:
        for line in f:
            m = CONFIG.match(line)
            if m and cfg is None:
                cfg = m.group(1)
            m = FINISHED.match(line)
            if m:
                status = int(m.group(1))
    return cfg, status


def _gzip(src: Path, dst: Path) -> bool:
    """Gzip ``src`` to ``dst``, unless ``dst`` already holds the same log (a new gzip header alone would
    show every collected log as modified in git). True if it was written."""
    if dst.exists() and gzip.decompress(dst.read_bytes()) == src.read_bytes():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fi, gzip.open(dst, "wb", compresslevel=9) as fo:
        shutil.copyfileobj(fi, fo)
    return True


def from_slurm(paths: list[Path]) -> None:
    files = []
    for p in paths:
        files += sorted(p.rglob("*.out")) if p.is_dir() else [p]
    best: dict[str, Path] = {}
    failed: dict[str, Path] = {}
    for f in sorted(files, key=lambda x: x.stat().st_mtime):
        cfg, status = _header(f)
        if cfg is None:
            continue
        (best if status == 0 else failed)[cfg] = f
    for cfg, f in sorted(best.items()):
        out = run_log(_scope_of(cfg), _fold_of(cfg)).with_suffix(".log.gz")
        if _gzip(f, out):
            print(f"{f.name:<32} -> {out.relative_to(REPO)}")
    for cfg in sorted(set(failed) - set(best)):
        print(f"no successful attempt: {cfg} (last: {failed[cfg].name})")


def local() -> None:
    for f in sorted(QUEUE_LOGS.rglob("*.log")):
        cfg, _ = _header(f)
        if cfg is None:
            continue
        scope, fold = _scope_of(cfg), _fold_of(cfg)
        if not _results_already(scope, fold, 1):
            print(f"no results yet, left as is: {f.relative_to(REPO)}")
            continue
        out = f.with_suffix(".log.gz")
        _gzip(f, out)
        f.unlink()
        print(f"gzipped {out.relative_to(REPO)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", type=Path, help="Slurm .out files or directories holding them")
    ap.add_argument("--local", action="store_true", help="gzip the finished logs of the local queue runner")
    args = ap.parse_args()
    if args.paths:
        from_slurm(args.paths)
    if args.local:
        local()
    if not args.paths and not args.local:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
