#!/usr/bin/env python3
"""Per-instance search trajectories from the run logs: size of the best solution against oracle calls.

Every improvement is logged as ``Found solution with size: N, calls: k`` after ``Initial solution size``;
the instance id comes from the evaluator's ``Evaluating instance with id``. For two runs over the same
instances this prints, per instance, the initial size, each method's final size and the calls of its last
improvement, and the calls each method needed to reach the other's final size, so that instances where
one method keeps descending slowly while the other converges early stand out.

    python scripts/selector_trajectories.py lab/output/queue_logs/proteins-sel_dce_lcls_fold0.log \\
        lab/output/queue_logs/proteins-sel_dce_lcls-sel_fold0.log --names vanilla selector
"""
from __future__ import annotations

import argparse
import re

INST = re.compile(r"Evaluating instance with id (\d+)")
INIT = re.compile(r"Initial solution size: (\d+)")
FOUND = re.compile(r"Found solution with size: (\d+), calls: (\d+)")
DONE = re.compile(r"Oracle calls(?: limit reached)?[:,]? ?(\d+)?|outer iteration|original: (\d+), final: (\d+)")


def parse(path: str) -> dict[str, dict]:
    out, cur = {}, None
    for line in open(path, errors="replace"):
        m = INST.search(line)
        if m:
            cur = {"init": None, "traj": []}
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
    args = ap.parse_args()
    A, B = parse(args.a), parse(args.b)
    ids = [i for i in A if i in B]
    # the last instance of a running log is incomplete
    ids = ids[: (args.complete or len(ids))]
    na, nb = args.names
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
