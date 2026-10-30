#!/usr/bin/env python3
"""Plain LBS against the selector arms, from the run logs and the per-instance dumps: anytime curves,
budget-matched comparison, where the calls go, where the GED comes from, and the hard instances.

Reads, for every arm, the gzipped run logs (``Found solution with size: N, calls: k`` after
``Initial solution size``, the ``actual --->`` lines of plain LBS, the generator calls replayed from the
cache) and the ``cf_per_instance`` dumps, and keeps the instances every arm finished. The selector arms
also contribute their monitor rows (``lab/output/selector_logs``). Sections:

  A  anytime: mean best size after b minimizer calls, every arm (an arm that stopped keeps its final size)
  B  plain LBS cut at the calls where each selector arm stopped, against that arm's final size
  C  what plain LBS gets from going on after its first outer iteration without improvement, and the calls
     every arm spends after its last improvement
  D  calls each arm needs to reach the other's final size
  E  improvements by kind ((-) sweep, (-o) overshoot, (=) swap, (+) add): count and GED removed
  F  monitor means: trials per removal success, prequential AUC, calls and successes per phase, overshoot
  G  the hard instances: no single removal works at the start (rescued by the fallback, or never improved)
  H  plain LBS removal level grid: drop sizes by |S| (step = int(|S| / max_neigh) + 1)
  I  untrained against random, paired

    python scripts/analyze_selector_arms.py --folds 0 1 2 3 4
    python scripts/analyze_selector_arms.py --arms lbs=proteins-sel_dce_lcls sel=proteins-sel_dce_lcls-sel \\
        --monitor sel=selector --folds 0
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from result_paths import open_log, scope_root  # noqa: E402

INST = re.compile(r"Evaluating instance with id (\d+)")
INIT = re.compile(r"Initial solution size: (\d+)")
FOUND = re.compile(r"\((-o|-|=|\+)\) Found solution with size: (\d+), calls: (\d+)")
DONE = re.compile(r"evaluated instance with id (\d+)")
GEN = re.compile(r"\((\d+) generator oracle calls added back\)")
RED = re.compile(r"actual ---> (\d+)")
MONITOR = "lab/output/selector_logs/selector_{tag}_{dataset}.csv"


def parse(scope: str, fold: int) -> dict[int, dict]:
    out, cur = {}, None
    for line in open_log(f"{scope}:{fold}"):
        m = INST.search(line)
        if m:
            cur = {"init": None, "traj": [], "events": [], "done": False, "gen": None}
            out[int(m.group(1))] = cur
            continue
        if cur is None:
            continue
        if DONE.search(line):
            cur["done"] = True
            cur = None
            continue
        m = GEN.search(line)
        if m:
            cur["gen"] = int(m.group(1))
            continue
        m = INIT.search(line)
        if m and cur["init"] is None:
            cur["init"] = int(m.group(1))
            continue
        m = FOUND.search(line)
        if m:
            cur["traj"].append((int(m.group(3)), int(m.group(2))))
            cur["events"].append(("F", m.group(1), int(m.group(3)), int(m.group(2))))
            continue
        if RED.search(line):
            cur["events"].append(("R",))
    return out


def dumps(scope: str, fold: int) -> dict[int, tuple]:
    out = {}
    for f in glob.glob(str(scope_root(scope) / "**" / "cf_per_instance" / f"fold_{fold}" / "cf_*.json"), recursive=True):
        d = json.load(open(f))
        m = d["metrics"]
        out[int(d["id"])] = (m["GraphEditDistance"], m["OracleCalls"], d["input"]["num_nodes"], len(d["input"]["edges"]))
    return out


def size_at(t: dict, k: int) -> int:
    s = t["init"]
    for c, sz in t["traj"]:
        if c <= k:
            s = min(s, sz)
    return s


def calls_to_reach(t: dict, size: float):
    if t["init"] <= size:
        return 0
    for c, sz in t["traj"]:
        if sz <= size:
            return c
    return None


def first_failed_iteration(ev: list):
    """Index of the first ``actual --->`` whose outer iteration improved nothing: the next event is another
    reduction, a (-) improvement (the next iteration's removal sweep) or the end. Plain-LBS logs only."""
    for i, e in enumerate(ev):
        if e[0] != "R":
            continue
        nxt = ev[i + 1] if i + 1 < len(ev) else None
        if nxt is None or nxt[0] == "R" or (nxt[0] == "F" and nxt[1] == "-"):
            return i
    return None


def paired(x, y, a: str, b: str, label: str) -> None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    p = wilcoxon(x, y).pvalue if (x != y).any() else 1.0
    print(f"  {label}: {a} {x.mean():.2f} vs {b} {y.mean():.2f}; {a} better/tie/worse {(x < y).sum()}/{(x == y).sum()}/{(x > y).sum()}; p {p:.2g}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", nargs="+", default=["lbs=proteins-sel_dce_lcls-v0", "sel=proteins-sel_dce_lcls-sel",
                                                  "untrained=proteins-sel_dce_lcls-sel-untrained", "random=proteins-sel_dce_lcls-sel-random"],
                    help="name=scope; the first is plain LBS")
    ap.add_argument("--monitor", nargs="+", default=["sel=selector", "untrained=selector-untrained", "random=selector-random"],
                    help="name=model_tag of the arms with a monitor CSV")
    ap.add_argument("--dataset", default="PROTEINS-084c6a1896e8a96ebacf7ff4868f1197")
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--max-neigh", type=int, default=20)
    ap.add_argument("--budgets", type=int, nargs="+", default=[100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 7500, 10000])
    args = ap.parse_args()
    arms = dict(a.split("=", 1) for a in args.arms)
    tags = dict(a.split("=", 1) for a in args.monitor)
    lbs = next(iter(arms))
    sel_arms = [a for a in arms if a != lbs]

    T = {a: {} for a in arms}
    D = {a: {} for a in arms}
    for a, s in arms.items():
        for f in args.folds:
            T[a].update({(f, i): t for i, t in parse(s, f).items()})
            D[a].update({(f, i): d for i, d in dumps(s, f).items()})
    keys = sorted(k for k in D[lbs] if all(k in D[a] and k in T[a] and T[a][k]["done"] for a in arms))
    print(f"instances finished by every arm (folds {args.folds}): {len(keys)}")
    gen = {}
    for k in keys:
        g = [T[a][k]["gen"] for a in arms if T[a][k]["gen"] is not None]
        assert g and all(x == g[0] for x in g), (k, g)
        gen[k] = g[0]
    kmin = {a: {k: D[a][k][1] - gen[k] for k in keys} for a in arms}        # minimizer calls
    final = {a: {k: D[a][k][0] for k in keys} for a in arms}
    init = {k: T[lbs][k]["init"] for k in keys}
    for a in arms:
        assert all(T[a][k]["init"] == init[k] for k in keys), f"{a}: initial sizes differ from {lbs}"
        assert all(size_at(T[a][k], 10 ** 9) == final[a][k] for k in keys), f"{a}: logs and dumps disagree"
    mon = {}
    for a, tag in tags.items():
        mon[a] = {}
        for r in csv.DictReader(open(MONITOR.format(tag=tag, dataset=args.dataset))):
            mon[a][(int(r["fold"]), int(r["instance_id"]))] = r
        missing = [k for k in keys if k not in mon[a]]
        assert not missing, f"{a}: {len(missing)} instances without a monitor row"

    print("\n=== A. anytime: mean best size after b minimizer calls (an arm that has stopped keeps its final size)")
    print("b        " + "".join(f"{a:>11s}" for a in arms) + "   | share still running at b: " + " ".join(arms))
    for b in args.budgets:
        row = [np.mean([size_at(T[a][k], b) for k in keys]) for a in arms]
        run = [np.mean([kmin[a][k] > b for k in keys]) for a in arms]
        print(f"{b:<8d} " + "".join(f"{v:11.2f}" for v in row) + "   | " + " ".join(f"{v:5.2f}" for v in run))

    print(f"\n=== B. plain LBS ({lbs}) cut at the calls where each selector arm stopped, per instance")
    for a in sel_arms:
        cut = np.array([size_at(T[lbs][k], kmin[a][k]) for k in keys], float)
        fin = np.array([final[a][k] for k in keys], float)
        print(f"  {a:10s} final {fin.mean():6.2f} | LBS at the same calls {cut.mean():6.2f} | LBS with its full budget "
              f"{np.mean([final[lbs][k] for k in keys]):6.2f} | {a} better/tie/worse than cut LBS {(fin < cut).sum()}/{(fin == cut).sum()}/{(fin > cut).sum()},"
              f" p {wilcoxon(fin, cut).pvalue:.1e}")

    print("\n=== C. plain LBS after its first outer iteration without improvement; calls after the last improvement")
    late, n_late, then = [], 0, []
    for k in keys:
        ev = T[lbs][k]["events"]
        i = first_failed_iteration(ev)
        s = final[lbs][k] if i is None else min([e[3] for e in ev[:i] if e[0] == "F"], default=init[k])
        late.append(s - final[lbs][k]); n_late += s > final[lbs][k]; then.append(s)
    descent = sum(init[k] - final[lbs][k] for k in keys)
    print(f"  instances improving after a fully failed outer iteration: {n_late}/{len(keys)}; GED gained there: mean {np.mean(late):.2f}"
          f" (sum {sum(late):.0f} of LBS's total descent {descent:.0f}); mean size at that point {np.mean(then):.2f}, final {np.mean([final[lbs][k] for k in keys]):.2f}")
    for a in arms:
        kl = np.array([T[a][k]["traj"][-1][0] if T[a][k]["traj"] else 0 for k in keys]); kt = np.array([kmin[a][k] for k in keys])
        print(f"  {a:10s} calls: total mean {kt.mean():.0f}; last improvement at mean {kl.mean():.0f} (median {np.median(kl):.0f});"
              f" spent after it: mean {(kt - kl).mean():.0f} = {(kt - kl).sum() / kt.sum() * 100:.0f} %")
    for a in arms:
        half = []
        for k in keys:
            c = calls_to_reach(T[a][k], init[k] - 0.5 * (init[k] - final[a][k]))
            half.append(c if c is not None else kmin[a][k])
        firsts = [T[a][k]["traj"][0][0] for k in keys if T[a][k]["traj"]]
        print(f"  {a:10s} first improvement: median {np.median(firsts):.0f} calls (mean {np.mean(firsts):.0f});"
              f" half of its own descent: median {np.median(half):.0f} (mean {np.mean(half):.0f})")

    print("\n=== D. calls to reach the other arm's final size")
    for a in sel_arms:
        r1 = [calls_to_reach(T[a][k], final[lbs][k]) for k in keys]
        r2 = [calls_to_reach(T[lbs][k], final[a][k]) for k in keys]
        ok1 = [x for x in r1 if x is not None]; ok2 = [x for x in r2 if x is not None]
        print(f"  {a:10s} reaches LBS's final size in {len(ok1)}/{len(keys)} (median {np.median(ok1):.0f} calls);"
              f" LBS reaches {a}'s in {len(ok2)}/{len(keys)} (median {np.median(ok2):.0f})")

    print("\n=== E. improvements by kind: count, GED removed (share), mean drop")
    for a in arms:
        agg = {}
        for k in keys:
            prev = init[k]
            for e in T[a][k]["events"]:
                if e[0] == "F":
                    d = agg.setdefault(e[1], [0, 0]); d[0] += 1; d[1] += prev - e[3]; prev = e[3]
        tot = sum(v[1] for v in agg.values()) or 1
        print(f"  {a:10s} " + "  ".join(f"({kd}) n={v[0]} ged={v[1]} ({v[1] / tot * 100:.0f}%) mean={v[1] / v[0]:.1f}" for kd, v in sorted(agg.items())))

    if mon:
        print("\n=== F. monitor means per instance (rem_tries_mean = sweep trials per removal success; rem_auc prequential, cascade-biased)")
        cols = ["rem_n", "rem_auc", "rem_tries_mean", "rem_successes", "calls_remove", "calls_swap", "calls_add",
                "succ_remove", "succ_swap", "succ_add", "wall_s"]
        print("            " + "".join(f"{c:>14s}" for c in cols) + "   overshoot calls / successes / mean size")
        for a in mon:
            vals = [np.nanmean([float(mon[a][k][c]) if mon[a][k][c] not in ("", "nan") else np.nan for k in keys]) for c in cols]
            oc, osz = [], []
            for k in keys:
                o = json.loads(mon[a][k]["overshoot"]); oc.append(o["calls"]); osz += o["successes"]
            print(f"  {a:10s}" + "".join(f"{v:14.3f}" for v in vals) + f"   {np.mean(oc):.0f} / {len(osz)} / {np.mean(osz) if osz else 0:.1f}")

    ref = sel_arms[0]
    print(f"\n=== G. hard instances: no single removal worked at the start of {ref} (first improvement by the fallback, or never)")
    fb = [k for k in keys if T[ref][k]["traj"] and T[ref][k]["events"][0][1] == "-o"]
    never = [k for k in keys if not T[ref][k]["traj"]]
    rest = [k for k in keys if k not in set(never)]
    print(f"  rescued by the fallback {len(fb)} (init mean {np.mean([init[k] for k in fb]) if fb else 0:.1f}): final "
          + ", ".join(f"{a} {np.mean([final[a][k] for k in fb]) if fb else 0:.1f}" for a in arms))
    print(f"  never improved {len(never)} (init mean {np.mean([init[k] for k in never]) if never else 0:.1f}, init/edges "
          f"{np.mean([init[k] / max(1, D[ref][k][3]) for k in never]) if never else 0:.2f}): improved by " + ", ".join(f"{a} {sum(final[a][k] < init[k] for k in never)}" for a in arms))
    for a in arms:
        s = sum(final[a][k] for k in never); r = sum(final[a][k] for k in rest)
        print(f"    {a:10s} GED sum: never-improved {s:.0f} ({s / max(1, s + r) * 100:.0f} % of its total {s + r:.0f}), rest {r:.0f} (mean {r / max(1, len(rest)):.2f});"
              f" calls on the never-improved mean {np.mean([kmin[a][k] for k in never]) if never else 0:.0f}")
    if mon and never:
        for a in mon:
            print(f"    {a:10s} calls by phase on the never-improved: " + ", ".join(f"{c} {np.mean([float(mon[a][k][c]) for k in never]):.0f}" for c in ("calls_remove", "calls_swap", "calls_add"))
                  + f", overshoot {np.mean([json.loads(mon[a][k]['overshoot'])['calls'] for k in never]):.0f}")
    drops = [(init[k] - T[lbs][k]["traj"][0][1], T[lbs][k]["traj"][0][0]) for k in never if T[lbs][k]["traj"]]
    print(f"  LBS's first improvement on the never-improved: {drops}")
    print(f"  on the rest ({len(rest)}), paired against {ref}:")
    for a in sel_arms[1:]:
        paired([final[ref][k] for k in rest], [final[a][k] for k in rest], ref, a, "GED")
        paired([kmin[ref][k] for k in rest], [kmin[a][k] for k in rest], ref, a, "calls")

    print(f"\n=== H. plain LBS (-) drop sizes by |S| before the move (step = int(|S| / {args.max_neigh}) + 1)")
    by = {}
    for k in keys:
        prev = init[k]
        for e in T[lbs][k]["events"]:
            if e[0] == "F":
                if e[1] == "-":
                    by.setdefault(prev // args.max_neigh, Counter())[prev - e[3]] += 1
                prev = e[3]
    for band in sorted(by):
        c = by[band]; n = sum(c.values())
        print(f"  |S| in [{band * args.max_neigh}, {(band + 1) * args.max_neigh}): n={n:5d}  " + "  ".join(f"drop={d}: {c[d] / n * 100:5.1f}%" for d in range(1, 5))
              + f"  drop>=5: {sum(v for d, v in c.items() if d >= 5) / n * 100:5.1f}%")
    for a in sel_arms:
        c = Counter()
        for k in keys:
            prev = init[k]
            for e in T[a][k]["events"]:
                if e[0] == "F":
                    if e[1] == "-" and prev - e[3] == 1:
                        c[prev >= args.max_neigh] += 1
                    prev = e[3]
        print(f"  {a:10s} single-edge successes: |S| < {args.max_neigh}: {c[False]}, |S| >= {args.max_neigh}: {c[True]}")

    if len(sel_arms) >= 3:
        print(f"\n=== I. {sel_arms[1]} against {sel_arms[2]}, paired")
        paired([final[sel_arms[1]][k] for k in keys], [final[sel_arms[2]][k] for k in keys], sel_arms[1], sel_arms[2], "GED")
        paired([kmin[sel_arms[1]][k] for k in keys], [kmin[sel_arms[2]][k] for k in keys], sel_arms[1], sel_arms[2], "calls")
        if mon:
            for a, b in ((sel_arms[0], sel_arms[1]), (sel_arms[0], sel_arms[2]), (sel_arms[1], sel_arms[2])):
                paired([float(mon[a][k]["wall_s"]) for k in keys], [float(mon[b][k]["wall_s"]) for k in keys], a, b, "minimizer wall s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
