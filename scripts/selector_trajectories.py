#!/usr/bin/env python3
"""Per-instance search trajectories from the run logs: size of the best solution against oracle calls.

Every improvement is logged as ``Found solution with size: N, calls: k`` after ``Initial solution size``;
the instance id comes from the evaluator's ``Evaluating instance with id``. For two runs over the same
instances this prints, per instance, the initial size, each method's final size and the calls of its last
improvement, and the calls each method needed to reach the other's final size, so that instances where
one method keeps descending slowly while the other converges early stand out.

    python scripts/selector_trajectories.py proteins-sel_dce_lcls:0 proteins-sel_dce_lcls-sel:0 \\
        --names vanilla selector      # or the log paths (lab/output/queue_logs/<dataset>/<gen>-<min>/fold<f>.log)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from result_paths import open_log  # noqa: E402

INST = re.compile(r"Evaluating instance with id (\d+)")
INIT = re.compile(r"Initial solution size: (\d+)")
FOUND = re.compile(r"Found solution with size: (\d+), calls: (\d+)")
DONE = re.compile(r"evaluated instance with id (\d+)")


def parse(path: str) -> dict[str, dict]:
    out, cur = {}, None
    for line in open_log(path):
        m = DONE.search(line)
        if m and m.group(1) in out:
            out[m.group(1)]["done"] = True
            continue
        m = INST.search(line)
        if m:
            cur = {"init": None, "traj": [], "done": False}
            out[m.group(1)] = cur
            continue
        if cur is None:
            continue
        m = INIT.search(line)
        if m and cur["init"] is None:
            cur["init"] = int(m.group(1))
            continue
        m = FOUND.search(line)
        if m:
            cur["traj"].append((int(m.group(2)), int(m.group(1))))
    return out


def final(t: dict) -> tuple[int, int]:
    """(final size, calls of the last improvement)."""
    return (t["traj"][-1][1], t["traj"][-1][0]) if t["traj"] else (t["init"], 0)


def calls_to_reach(t: dict, size: int) -> int | None:
    if t["init"] is not None and t["init"] <= size:
        return 0
    for k, s in t["traj"]:
        if s <= size:
            return k
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--complete", type=int, default=None, help="only the first N instances of each log")
    ap.add_argument("--summary", action="store_true", help="category counts and the stalled-A instances only")
    args = ap.parse_args()
    A, B = parse(args.a), parse(args.b)
    ids = [i for i in A if i in B and A[i].get("done") and B[i].get("done")]
    ids = ids[: (args.complete or len(ids))]
    na, nb = args.names
    if args.summary:
        better = [i for i in ids if final(B[i])[0] < final(A[i])[0]]
        worse = [i for i in ids if final(B[i])[0] > final(A[i])[0]]
        # A "did not get down in time": B ends clearly lower (at least 3 edges and 20 %), or B reaches A's
        # final size with under a tenth of the calls A needed for it
        stalled = []
        for i in ids:
            fa, ka = final(A[i]); fb, kb = final(B[i]); r = calls_to_reach(B[i], fa)
            if (fa - fb >= 3 and fb <= 0.8 * fa) or (r is not None and ka >= 500 and r <= 0.1 * ka):
                stalled.append((i, A[i]["init"], fa, ka, fb, kb, r))
        print(f"{len(ids)} instances finished by both: {nb} better {len(better)}, equal {len(ids) - len(better) - len(worse)}, "
              f"worse {len(worse)}" + (f" ({', '.join(worse)})" if worse else ""))
        print(f"\n{na} did not get down in time, {nb} did ({len(stalled)}):")
        print(f"{'id':>6s}{'init':>6s} | {na+' final @calls':>22s} | {nb+' final @calls':>22s} | {nb+' reaches '+na+' final at':>28s}")
        for i, ini, fa, ka, fb, kb, r in stalled:
            print(f"{i:>6s}{ini:>6d} | {fa:>12d} @ {ka:<7d} | {fb:>12d} @ {kb:<7d} | {str(r):>28s}")
        return 0
    print(f"{'id':>6s}{'init':>6s} | {na+' final':>13s}{'@calls':>8s} | {nb+' final':>14s}{'@calls':>8s} | "
          f"{nb+' to reach '+na:>22s}")
    rows = []
    for i in ids:
        fa, ka = final(A[i]); fb, kb = final(B[i])
        reach = calls_to_reach(B[i], fa)
        rows.append((i, A[i]["init"], fa, ka, fb, kb, reach))
        print(f"{i:>6s}{A[i]['init']:>6d} | {fa:>13d}{ka:>8d} | {fb:>14d}{kb:>8d} | {str(reach):>22s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
