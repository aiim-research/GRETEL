#!/usr/bin/env python3
"""The protocol v2 fleet on DCEM (docs/v2/RUNS.md): the selector paper's ladder per cell, pooled over the folds
every arm of the cell has finished.

Per cell and arm: instances, correctness, and over the correct instances the mean structural cost
(StructuralEditDistance), feature cost (MatchedFeatureDistance), total (their sum) and oracle calls. Then the
paired comparisons up the ladder on the total cost, over the instances correct in both arms (mean difference,
wins/ties/losses, Wilcoxon p): bls_v0 against OBS and RHC, bls_v1-F against bls_v0, selector_v1-F against
bls_v1-F and against the untrained control-F. With --paper1, paper 1's main table (DCEM alone, with BLS, with
OBS; structural GED) next to the published 10-fold means.

    python scripts/v2_fleet.py [--folds 0-8] [--paper1] [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "lab" / "output" / "results"
KEYS = {"struct": "StructuralEditDistance", "feat": "MatchedFeatureDistance", "calls": "OracleCalls", "ok": "Correctness"}
ARMS = [("DCEM", "dummy"), ("OBS", "obs"), ("RHC", "rhc"), ("bls_v0", "lcls-v0"), ("bls_v1-F", "lcls-v1-f"),
        ("control-F", "lcls-sel-v1-untrained-f"), ("selector_v1-F", "lcls-sel-v1-f")]
PAIRS = [("bls_v0", "OBS"), ("bls_v0", "RHC"), ("bls_v1-F", "bls_v0"), ("selector_v1-F", "bls_v1-F"),
         ("selector_v1-F", "control-F"), ("control-F", "bls_v1-F")]
# Paper 1, Table dce_vs_dcem (10-fold means, protocol v1): DCEM alone, DCEM-LBS, DCEM-OBS, structural GED.
PAPER1 = {"synthie": (201.41, 2.16, 185.22), "bbbp": (18.98, 3.20, 14.43), "enzymes": (58.46, 2.55, 14.46),
          "bzr": (38.06, 2.06, 28.79), "aids": (17.25, 13.02, 16.92), "proteins": (73.19, 50.18, 64.58)}


def load(scope: str, folds: set[int]) -> dict[tuple[int, str], dict[str, float]]:
    out = {}
    for f in glob.glob(str(RESULTS / scope / "**" / "results_*_*.json"), recursive=True):
        fold = int(re.search(r"results_(\d+)_", f).group(1))
        if fold not in folds:
            continue
        res = json.loads(Path(f).read_text())["results"]
        for short, key in KEYS.items():
            full = next((k for k in res if k.endswith("." + key)), None)
            for e in res.get(full, []):
                out.setdefault((fold, str(e["id"])), {})[short] = float(e["value"])
    for v in out.values():
        v["total"] = v.get("struct", 0.0) + v.get("feat", 0.0)
    return out


def folds_of(scope: str) -> set[int]:
    return {int(re.search(r"results_(\d+)_", f).group(1))
            for f in glob.glob(str(RESULTS / scope / "**" / "results_*_*.json"), recursive=True)}


def paired(a: dict, b: dict, key: str = "total"):
    ids = [i for i in a if i in b and a[i].get("ok", 0) >= 1 and b[i].get("ok", 0) >= 1]
    if not ids:
        return None
    x = np.array([a[i][key] for i in ids]); y = np.array([b[i][key] for i in ids]); d = x - y
    try:
        p = float(wilcoxon(x, y).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    return {"n": len(ids), "diff": float(d.mean()), "rel": float(d.mean() / max(y.mean(), 1e-9)),
            "better": int((d < 0).sum()), "ties": int((d == 0).sum()), "worse": int((d > 0).sum()), "p": p}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default=None, help="e.g. 0-8; default: the folds every arm of the cell has")
    ap.add_argument("--paper1", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    cells = sorted({Path(p).name.split("-sel_dcm_")[0].removeprefix("v2-") for p in RESULTS.glob("v2-*-sel_dcm_dummy")})
    report = {}
    for cell in cells:
        if args.folds:
            lo, hi = map(int, args.folds.split("-"))
            folds = set(range(lo, hi + 1))
        else:
            sets = [folds_of(f"v2-{cell}-sel_dcm_{s}") for _, s in ARMS]
            sets = [s for s in sets if s]
            folds = set.intersection(*sets) if sets else set()
        data = {name: load(f"v2-{cell}-sel_dcm_{s}", folds) for name, s in ARMS}
        rows = {}
        for name, d in data.items():
            ok = [v for v in d.values() if v.get("ok", 0) >= 1]
            if not d:
                continue
            rows[name] = {"n": len(d), "ok": len(ok) / len(d), **{k: float(np.mean([v[k] for v in ok])) if ok else float("nan")
                                                                  for k in ("struct", "feat", "total", "calls")}}
        pairs = {f"{a} vs {b}": paired(data[a], data[b]) for a, b in PAIRS if data.get(a) and data.get(b)}
        report[cell] = {"folds": sorted(folds), "rows": rows, "pairs": pairs}
        print(f"\n== {cell}  (folds {min(folds) if folds else '-'}-{max(folds) if folds else '-'}, {len(folds)})")
        print(f"   {'arm':14s} {'n':>5s} {'ok':>5s} {'struct':>8s} {'feat':>8s} {'total':>8s} {'calls':>7s}")
        for name, r in rows.items():
            print(f"   {name:14s} {r['n']:5d} {r['ok']:5.2f} {r['struct']:8.2f} {r['feat']:8.2f} {r['total']:8.2f} {r['calls']:7.0f}")
        for k, r in pairs.items():
            if r:
                print(f"   {k:28s} total {r['diff']:+8.2f} ({r['rel']:+.0%})  {r['better']:4d}<{r['ties']:4d}={r['worse']:4d}>  p {r['p']:.0e}")
    if args.paper1:
        print(f"\nPaper 1 main table on DCEM (structural GED): published 10-fold mean / v2")
        for ds, (g, l, o) in PAPER1.items():
            r = report.get(ds, {}).get("rows", {})
            v = lambda n: r.get(n, {}).get("struct", float("nan"))
            print(f"   {ds:9s} DCEM {g:7.2f} / {v('DCEM'):7.2f}   BLS {l:6.2f} / {v('bls_v0'):6.2f}   OBS {o:7.2f} / {v('OBS'):7.2f}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
