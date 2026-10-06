#!/usr/bin/env python3
"""Final size and minimizer calls per instance from several run logs (one column per arm), next to plain
LBS from its per-instance dumps. For short preliminary runs whose results are only in their logs.

    python scripts/compare_prelim_logs.py --vanilla proteins-sel_dce_lcls --fold 0 \\
        ref=/path/ref.log p1=/path/p1.log p2=/path/p2.log p12=/path/p12.log
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_instances import load  # noqa: E402


def parse(path: str) -> tuple[list[str], dict[str, tuple[int, int]]]:
    order, out, cur = [], {}, None
    if not Path(path).exists():
        return order, out
    for line in open(path, errors="replace"):
        m = re.search(r"Evaluating instance with id (\d+)$", line.strip())
        if m:
            cur = m.group(1); order.append(cur); continue
        m = re.search(r"Oracle calls: (\d+)$", line.strip())
        if m and cur:
            calls = int(m.group(1))
        m = re.search(r"original: (\d+), final: (\d+)", line)
        if m and cur:
            out[cur] = (int(m.group(2)), calls)
    return order, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("arms", nargs="+", help="name=log"); ap.add_argument("--vanilla", required=True)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    arms = dict(a.split("=", 1) for a in args.arms)
    data = {n: parse(p) for n, p in arms.items()}
    van = load(args.vanilla, args.fold)
    ids = max((d[0] for d in data.values()), key=len)
    print("| id | vanilla GED/calls | " + " | ".join(f"{n} GED/calls" for n in arms) + " |")
    print("|---|---|" + "---|" * len(arms))
    tot = {n: [0, 0] for n in arms}; tv = 0
    for i in ids:
        v = van.get(i)
        cells = []
        for n in arms:
            r = data[n][1].get(i)
            cells.append(f"{r[0]} / {r[1]:,}" if r else "...")
            if r and v:
                tot[n][0] += r[1]; tot[n][1] += 1
        if v and all(data[n][1].get(i) for n in arms):
            tv += int(v["oc"]) - 1121
        print(f"| {i} | {int(v['ged'])} / {int(v['oc']) - 1121:,} | " + " | ".join(cells) + " |" if v else f"| {i} | ? | " + " | ".join(cells) + " |")
    print("\ntotal minimizer calls over the instances finished by every arm: vanilla "
          f"{tv:,}; " + "; ".join(f"{n} {tot[n][0]:,}" for n in arms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
