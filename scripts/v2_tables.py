#!/usr/bin/env python3
"""Protocol v2 tables: one row per scope (dataset x generator x minimizer) from the results JSON of a fold.

Per scope: instances, correctness, and over the correct instances the mean structural cost
(StructuralEditDistance), feature cost (MatchedFeatureDistance), oracle calls (generation included) and
runtime. With --pairs, paired comparisons on the instances correct in both scopes (mean difference,
wins/ties/losses and the Wilcoxon p value) for the generator pairs (DCE against DCEM, OFS, DFS under the
same minimizer) and the minimizer pairs (BLS against BLS-F, OBS, RHC from the same generator).

    python scripts/v2_tables.py [--fold 0] [--pairs] [--json out.json]
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
KEYS = {"struct": "StructuralEditDistance", "feat": "MatchedFeatureDistance", "calls": "OracleCalls",
        "time": "Runtime", "ok": "Correctness"}


def load(scope: str, fold: int) -> dict[str, dict[str, float]] | None:
    files = glob.glob(str(RESULTS / scope / "**" / f"results_{fold}_*.json"), recursive=True)
    if not files:
        return None
    res = json.loads(Path(sorted(files)[-1]).read_text())["results"]
    out: dict[str, dict[str, float]] = {}
    for short, key in KEYS.items():
        full = next((k for k in res if k.endswith("." + key)), None)
        for e in res.get(full, []):
            out.setdefault(str(e["id"]), {})[short] = float(e["value"])
    return out


def summary(inst: dict[str, dict[str, float]]) -> dict[str, float]:
    ok = [v for v in inst.values() if v.get("ok", 0) >= 1]
    s = {"n": len(inst), "ok": len(ok) / max(len(inst), 1)}
    for k in ("struct", "feat", "calls", "time"):
        vals = [v[k] for v in ok if k in v]
        s[k] = float(np.mean(vals)) if vals else float("nan")
    return s


def paired(a: dict, b: dict, key: str = "struct") -> dict[str, float] | None:
    ids = [i for i in a if i in b and a[i].get("ok", 0) >= 1 and b[i].get("ok", 0) >= 1]
    if not ids:
        return None
    x = np.array([a[i][key] for i in ids])
    y = np.array([b[i][key] for i in ids])
    d = y - x
    try:
        p = float(wilcoxon(x, y).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    return {"n": len(ids), "a": float(x.mean()), "b": float(y.mean()), "diff": float(d.mean()),
            "b_better": int((d < 0).sum()), "ties": int((d == 0).sum()), "b_worse": int((d > 0).sum()), "p": p}


def scopes(fold: int) -> dict[str, dict]:
    out = {}
    for d in sorted(RESULTS.glob("v2-*")):
        name = d.name
        if "smoke" in name:
            continue
        inst = load(name, fold)
        if inst:
            out[name[3:]] = inst
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--pairs", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    data = scopes(args.fold)
    rows = {s: summary(v) for s, v in data.items()}
    print(f"{'scope':42s} {'n':>4s} {'ok':>5s} {'struct':>7s} {'feat':>7s} {'calls':>8s} {'time':>7s}")
    for s, r in rows.items():
        print(f"{s:42s} {r['n']:4d} {r['ok']:5.2f} {r['struct']:7.2f} {r['feat']:7.2f} {r['calls']:8.0f} {r['time']:7.1f}")
    pairs = {}
    if args.pairs:
        for s in data:
            m = re.fullmatch(r"(.+)_(dce)_(\w[\w.-]*)", s)
            if not m:
                continue
            ds, _, mini = m.groups()
            for gen in ("dcm", "ofs", "dfs"):
                o = f"{ds}_{gen}_{mini}"
                if o in data:
                    pairs[f"{s} vs {gen}"] = paired(data[s], data[o])
        for s in data:
            m = re.fullmatch(r"(.+)_(\w+)_lcls", s)
            if not m:
                continue
            for mini in ("lclsf", "obs", "rhc"):
                o = f"{m.group(1)}_{m.group(2)}_{mini}"
                if o in data:
                    pairs[f"{s} vs {mini}"] = paired(data[s], data[o])
        print(f"\n{'pair (b against a, structural cost)':52s} {'n':>4s} {'a':>7s} {'b':>7s} {'b-a':>7s} {'b<a':>4s} {'=':>4s} {'b>a':>4s} {'p':>8s}")
        for k, r in pairs.items():
            if r:
                print(f"{k:52s} {r['n']:4d} {r['a']:7.2f} {r['b']:7.2f} {r['diff']:7.2f} {r['b_better']:4d} {r['ties']:4d} {r['b_worse']:4d} {r['p']:8.1e}")
    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows, "pairs": pairs}, indent=1))


if __name__ == "__main__":
    main()
