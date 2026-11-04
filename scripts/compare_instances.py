#!/usr/bin/env python3
"""Per-instance GED / oracle calls of two runs, from the per-instance dumps written as each instance ends.

Reads ``cf_per_instance/fold_<f>/cf_<id>.json`` under each scope (available while the run is still
going), keeps the instances both runs finished, and prints GED/OC for each, grouped: B ends with a lower
GED, same GED, B ends with a higher GED. OracleCalls is the total of the instance as the pipeline counts
it, which includes the generator's calls (identical for both runs, since the generator output is shared).
With ``--logs`` it also shows, per method, the minimizer calls spent up to its last improvement (from the
``Found solution with size: N, calls: k`` lines): when the method stopped making progress.

    python scripts/compare_instances.py proteins-sel_dce_lcls proteins-sel_dce_lcls-sel --fold 0 \\
        --names vanilla selector
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from result_paths import scope_root  # noqa: E402
from selector_trajectories import parse, final  # noqa: E402


def load(scope: str, fold: int) -> dict[str, dict]:
    out = {}
    for f in glob.glob(str(scope_root(scope) / "**" / "cf_per_instance" / f"fold_{fold}" / "cf_*.json"), recursive=True):
        d = json.load(open(f))
        m = d["metrics"]
        out[d["id"]] = {"ged": m.get("GraphEditDistance"), "oc": m.get("OracleCalls")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--logs", nargs=2, default=None, help="run logs of A and B (paths or <scope>:<fold>), for the calls up to the last improvement")
    args = ap.parse_args()
    A, B = load(args.a, args.fold), load(args.b, args.fold)
    last = [{}, {}]
    if args.logs:
        for j, lp in enumerate(args.logs):
            last[j] = {i: final(t)[1] for i, t in parse(lp).items() if t.get("init") is not None}
    ids = sorted(set(A) & set(B), key=int)
    na, nb = args.names
    groups = {"lower": [], "same": [], "higher": []}
    for i in ids:
        ga, gb = A[i]["ged"], B[i]["ged"]
        groups["lower" if gb < ga else "same" if gb == ga else "higher"].append(i)
    oa = sum(A[i]["oc"] for i in ids); ob = sum(B[i]["oc"] for i in ids)
    print(f"{len(ids)} instances finished by both. {nb} lower GED: {len(groups['lower'])}, same: {len(groups['same'])}, "
          f"higher: {len(groups['higher'])}. Total OC {na} {oa:,} vs {nb} {ob:,} ({100 * (ob - oa) / oa:+.0f} %).")
    titles = {"lower": f"{nb} ends with a lower GED", "same": "same GED", "higher": f"{nb} ends with a higher GED"}
    for g in ("lower", "same", "higher"):
        if not groups[g]:
            continue
        print(f"\n{titles[g]} ({len(groups[g])}):")
        if args.logs:
            print(f"| id | {na} GED/OC | {na} last improvement at | {nb} GED/OC | {nb} last improvement at |")
            print("|---|---|---|---|---|")
            for i in groups[g]:
                la, lb = (f"{last[j][i]:,}" if i in last[j] else "?" for j in (0, 1))
                print(f"| {i} | {A[i]['ged']:.0f} / {A[i]['oc']:,} | {la} | {B[i]['ged']:.0f} / {B[i]['oc']:,} | {lb} |")
        else:
            print(f"| id | {na} GED/OC | {nb} GED/OC |")
            print("|---|---|---|")
            for i in groups[g]:
                print(f"| {i} | {A[i]['ged']:.0f} / {A[i]['oc']:,} | {B[i]['ged']:.0f} / {B[i]['oc']:,} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
