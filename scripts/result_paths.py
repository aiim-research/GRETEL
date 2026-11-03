"""Where the analysis scripts find results, selector monitors and run logs.

Live results are under ``lab/output/results/<scope>``. The selector development study is archived by
phase under ``lab/selector_experiments/<phase>/{results,selector_logs,queue_logs}`` (see its README), so a
bare scope name is looked up in ``lab/output/results`` first and then in every archived phase: the
analysis commands written before the archive keep working. A scope containing "/" is a path relative to
the repository root, or else to ``lab/output``. Logs may be gzipped, and given as ``<scope>:<fold>``.

Run logs are grouped in one folder per dataset, generator and minimizer, taken from the scope
(``<dataset>_<generator>_<minimizer>``, e.g. ``proteins-sel_dce_lcls-sel`` -> ``proteins-sel/dce-lcls-sel``):
``lab/output/queue_logs/<dataset>/<generator>-<minimizer>/fold<f>.log`` for the local queue runner, and
``output/logs/<dataset>/<generator>-<minimizer>/fold<f>_<job>_<task>.out`` for Slurm jobs.
"""
from __future__ import annotations

import gzip
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "lab" / "output" / "results"
MONITORS = REPO / "lab" / "output" / "selector_logs"
ARCHIVE = REPO / "lab" / "selector_experiments"
QUEUE_LOGS = REPO / "lab" / "output" / "queue_logs"


def combo_dir(scope: str) -> Path:
    """``<dataset>/<generator>-<minimizer>`` for a scope named ``<dataset>_<generator>_<minimizer>``
    (also ``<dataset>_<generator>-<minimizer>``, as some early scopes are); the scope itself otherwise."""
    parts = scope.split("_")
    if len(parts) == 3:
        return Path(parts[0]) / f"{parts[1]}-{parts[2]}"
    if len(parts) == 2 and "-" in parts[1]:
        return Path(parts[0]) / parts[1]
    return Path(scope)


def run_log(scope: str, fold: int, root: Path = QUEUE_LOGS) -> Path:
    """Where the local queue runner writes the log of one config (``root`` for an archived phase)."""
    return root / combo_dir(scope) / f"fold{fold}.log"


def scope_root(scope: str) -> Path:
    if "/" in scope:
        for base in (REPO, REPO / "lab" / "output"):
            if (base / scope).exists():
                return base / scope
        return REPO / "lab" / "output" / scope
    for root in [RESULTS] + sorted(ARCHIVE.glob("*/results")):
        if (root / scope).exists():
            return root / scope
    return RESULTS / scope


def monitor_files(tag: str, dataset: str) -> list[Path]:
    pattern = f"selector_{tag}_{dataset}-*.csv"
    for d in [MONITORS] + sorted(ARCHIVE.glob("*/selector_logs")):
        files = sorted(d.glob(pattern))
        if files:
            return files
    return []


def resolve_log(path: str) -> Path | None:
    """A run log given as a path (``x.log`` also finds ``x.log.gz``) or as ``<scope>:<fold>``, looked up
    in ``lab/output/queue_logs`` and then in every archived phase. None if there is none."""
    for p in (Path(path), Path(f"{path}.gz")):
        if p.exists():
            return p
    if ":" in path and "/" not in path:
        scope, fold = path.rsplit(":", 1)
        for root in [QUEUE_LOGS] + sorted(ARCHIVE.glob("*/queue_logs")):
            for p in (run_log(scope, int(fold), root), run_log(scope, int(fold), root).with_suffix(".log.gz")):
                if p.exists():
                    return p
    return None


def open_log(path: str):
    """Open a run log (see ``resolve_log``) for reading text, plain or gzipped."""
    p = resolve_log(path) or Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", errors="replace")
    return open(p, errors="replace")
