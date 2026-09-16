"""Aggregation pipeline for the result store, matching lab/stats_visualizer.ipynb.

The value a bar or table cell shows for a (dataset, generator, minimizer, metric)
combination is built in three steps:

  * filter: GED and FED keep only the correct counterfactual records,
    Oracle Calls and Correctness keep every record.
  * per-fold mean of the per-instance values.
  * displayed value = mean over folds.

Reads ``lab/output/results/<ds>_<gen>_<min>/`` and nothing else, so the numbers
come straight from what the pipeline wrote. Used by scripts/make_paper_figures.py.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import jsonpickle

REPO = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO / "lab" / "output" / "results"

METRICS = ['GraphEditDistance', 'FeatureEditDistance', 'OracleCalls', 'Correctness']
GED = 'src.evaluation.future.stages.ged.GraphEditDistance'
FED = 'src.evaluation.future.stages.fed.FeatureEditDistance'
OC = 'src.evaluation.future.stages.oracle_calls.OracleCalls'
CORR = 'src.evaluation.future.stages.correctness.Correctness'
STAGE_OF = {'GraphEditDistance': GED, 'FeatureEditDistance': FED,
            'OracleCalls': OC, 'Correctness': CORR}
CORRECTNESS_FILTERED = {'GraphEditDistance', 'FeatureEditDistance'}


def _as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def discover_files(ds, gen, min_kind):
    """Return the ``(path, fold, seed, store)`` entries backing one cell.

    The scope is the top-level directory name under results/, so glob that
    directory directly instead of scanning and decoding the whole tree."""
    out = []
    sdir = RESULTS_ROOT / f"{ds}_{gen}_{min_kind}"
    if not sdir.is_dir():
        return out
    for f in sdir.rglob('results_*.json'):
        mm = re.match(r'results_(\d+)_(\d+)\.json', f.name)
        if not mm:
            continue
        out.append((f, int(mm.group(1)), 0, 'results'))
    return out


def _fold_correct_ids(data):
    ids = set()
    for e in data['results'].get(CORR, []):
        if _as_float(e.get('value')) and float(e['value']) > 0.5:
            ids.add(str(e.get('id')))
    return ids


def fold_mean(data, metric):
    """Per-fold mean for a metric, applying the correctness filter.

    GED/FED are averaged over *correct* counterfactual records only. The
    correctness flag is matched to each metric record **by position** (the
    stage lists are written in the same record order), not by instance id:
    some datasets (e.g. tcr-tco-300) store several counterfactual records per
    instance, so an id-based filter would wrongly pull in a failed record's
    GED=0 placeholder of an instance whose other record succeeded. Falls back
    to id-based matching if the two stage lists are not length-aligned."""
    stage = STAGE_OF[metric]
    entries = data['results'].get(stage, [])
    if metric in CORRECTNESS_FILTERED:
        corr_entries = data['results'].get(CORR, [])
        if len(corr_entries) == len(entries):
            vals = [_as_float(e['value'])
                    for e, ce in zip(entries, corr_entries)
                    if _as_float(ce.get('value')) is not None
                    and float(ce['value']) > 0.5
                    and _as_float(e.get('value')) is not None]
        else:  # robustness fallback: align by id
            correct = _fold_correct_ids(data)
            vals = [_as_float(e['value']) for e in entries
                    if str(e.get('id')) in correct
                    and _as_float(e.get('value')) is not None]
    else:
        vals = [_as_float(e['value']) for e in entries if _as_float(e.get('value')) is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def displayed_mean(ds, gen, min_kind, metric, files=None):
    """The value the bar or table shows for this cell, and the fold count."""
    if files is None:
        files = discover_files(ds, gen, min_kind)
    fms = []
    for f, fold, seed, store in files:
        data = jsonpickle.decode(f.read_text())
        fm = fold_mean(data, metric)
        if fm is not None:
            fms.append(fm)
    if not fms:
        return float('nan'), 0
    return float(np.mean(fms)), len(fms)
