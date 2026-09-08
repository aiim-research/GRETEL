#!/usr/bin/env python3
"""Analyse selector runs: per-instance comparison across scopes, start-of-fold learning curves from
the run logs, and within-fold training efficiency from the prequential monitor CSV.

Examples (repo root)::

    python scripts/analyze_selector_runs.py compare \
        base=synthie_dce_lcls v1=synthie_dce_lcls-net v2=synthie_dce_lcls-net-v2 --folds 0 1

    python scripts/analyze_selector_runs.py logs lab/output/queue_logs/synthie_dce_lcls-net-v2_fold0.log \
        lab/output/queue_logs/synthie_dce_lcls-net-v2_fold1.log --block 10

    python scripts/analyze_selector_runs.py monitor lab/output/selector_logs/selector_v2_Synthie-<hash>.csv
"""
from __future__ import annotations

import argparse
import base64
import csv
import glob
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "lab" / "output" / "results"

METRICS = ["GraphEditDistance", "FeatureEditDistance", "Correctness", "OracleCalls", "Runtime"]


# --------------------------------------------------------------------------- results

def _val(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict) and "py/reduce" in v:      # jsonpickle'd numpy scalar
        try:
            raw = base64.b64decode(v["py/reduce"][1]["py/tuple"][1]["py/b64"])
            return float(np.frombuffer(raw, dtype="<f4" if len(raw) == 4 else "<f8")[0])
        except Exception:
            return float("nan")
    return float("nan")


def load_fold(scope: str, fold: int, run: int = 1) -> dict[str, dict[str, float]] | None:
    # a scope containing "/" is taken as a path relative to lab/output (e.g. the pre-calibration backup)
    root = (REPO / "lab" / "output" / scope) if "/" in scope else (RESULTS / scope)
    files = glob.glob(str(root / "**" / f"results_{fold}_{run}.json"), recursive=True)
    if not files:
        return None
    d = json.load(open(files[0]))["results"]
    out: dict[str, dict[str, float]] = {}
    for stage, rows in d.items():
        name = stage.split(".")[-1]
        for r in rows:
            out.setdefault(str(r["id"]), {})[name] = _val(r["value"])
    return out


def cmd_compare(args) -> int:
    scopes = dict(s.split("=", 1) for s in args.scopes)
    data = {name: {} for name in scopes}
    for name, scope in scopes.items():
        for f in args.folds:
            r = load_fold(scope, f)
            if r is None:
                print(f"  (missing) {scope} fold {f}", file=sys.stderr)
                continue
            for iid, m in r.items():
                data[name][(f, iid)] = m
    names = list(scopes)
    base = names[0]
    common = set(data[base])
    for n in names[1:]:
        common &= set(data[n])
    common = sorted(common)
    if not common:
        print("no common instances")
        return 1
    print(f"\n== Pooled over folds {args.folds}: {len(common)} common instances ==")
    hdr = f"{'metric':22s}" + "".join(f"{n:>12s}" for n in names)
    print(hdr)
    for met in METRICS:
        row = f"{met:22s}"
        for n in names:
            row += f"{np.nanmean([data[n][k][met] for k in common]):12.2f}"
        print(row)
    for met in ("OracleCalls", "GraphEditDistance"):
        print(f"\n{met} quantiles (25/50/75/max):")
        for n in names:
            q = np.nanpercentile([data[n][k][met] for k in common], [25, 50, 75, 100])
            print(f"  {n:6s} " + " ".join(f"{x:9.1f}" for x in q))
    print("\nPer-instance versus", base)
    for n in names[1:]:
        for met in ("GraphEditDistance", "OracleCalls"):
            a = np.array([data[base][k][met] for k in common])
            b = np.array([data[n][k][met] for k in common])
            better = int((b < a - 1e-9).sum()); tie = int((np.abs(a - b) <= 1e-9).sum())
            print(f"  {n:6s} {met:18s} better {better:3d}  tie {tie:3d}  worse {len(common)-better-tie:3d}")
    if len(names) >= 3:
        a, b = names[1], names[2]
        print(f"\nPer-instance {b} versus {a}")
        for met in ("GraphEditDistance", "OracleCalls"):
            x = np.array([data[a][k][met] for k in common]); y = np.array([data[b][k][met] for k in common])
            better = int((y < x - 1e-9).sum()); tie = int((np.abs(x - y) <= 1e-9).sum())
            print(f"  {met:18s} {b} better {better:3d}  tie {tie:3d}  worse {len(common)-better-tie:3d}")
    print("\n== Per fold ==")
    for f in args.folds:
        keys = [k for k in common if k[0] == f]
        if not keys:
            continue
        print(f"fold {f} ({len(keys)} inst.)  " + "  ".join(
            f"{n}: GED {np.nanmean([data[n][k]['GraphEditDistance'] for k in keys]):.2f} "
            f"OC {np.nanmean([data[n][k]['OracleCalls'] for k in keys]):.0f}" for n in names))
    if args.per_id:
        print("\nid  " + "  ".join(f"{n}(GED/OC)" for n in names))
        for k in common:
            print(f"{k[0]}:{k[1]:>4s} " + "  ".join(
                f"{data[n][k]['GraphEditDistance']:.2f}/{data[n][k]['OracleCalls']:.0f}" for n in names))
    return 0


# ------------------------------------------------------------------------------ logs

INIT_RE = re.compile(r"Initial solution size: (\d+)")
OC_RE = re.compile(r"Oracle calls: (\d+)")
FINAL_RE = re.compile(r"original: (\d+), final: (\d+)")


def parse_log(path: str) -> list[dict]:
    """Per-instance sequence in processing order: initial size, minimizer oracle calls, final size."""
    inst = []
    cur = {}
    for line in open(path, errors="replace"):
        m = INIT_RE.search(line)
        if m:
            cur = {"initial": int(m.group(1))}
            continue
        m = OC_RE.search(line)
        if m and cur:
            cur["oc"] = int(m.group(1))
            continue
        m = FINAL_RE.search(line)
        if m and cur:
            cur["final"] = int(m.group(2))
            inst.append(cur)
            cur = {}
    return inst


def cmd_logs(args) -> int:
    for path in args.logs:
        inst = parse_log(path)
        if not inst:
            print(f"{path}: no instances parsed")
            continue
        oc = np.array([i["oc"] for i in inst], float)
        fin = np.array([i["final"] for i in inst], float)
        ini = np.array([i["initial"] for i in inst], float)
        b = args.block
        print(f"\n== {Path(path).name}: {len(inst)} instances ==")
        print(f"  mean initial {ini.mean():.1f}  mean final {fin.mean():.2f}  mean minimizer OC {oc.mean():.0f}  median {np.median(oc):.0f}")
        print("  minimizer OC by blocks of %d: %s" % (b, [int(oc[i:i + b].mean()) for i in range(0, len(oc), b)]))
        print("  final size by blocks of %d:   %s" % (b, [round(float(fin[i:i + b].mean()), 2) for i in range(0, len(fin), b)]))
        print("  OC per removed edge by blocks: %s" % [round(float((oc[i:i + b] / np.maximum(1, ini[i:i + b] - fin[i:i + b])).mean()), 2) for i in range(0, len(oc), b)])
        k = min(args.first, len(oc))
        print(f"  first {k} instances: mean OC {oc[:k].mean():.0f}, mean final {fin[:k].mean():.2f}, OC/removed-edge {float((oc[:k] / np.maximum(1, ini[:k] - fin[:k])).mean()):.2f}")
    return 0


# --------------------------------------------------------------------------- monitor

def _f(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def _trend(y: np.ndarray) -> str:
    """Spearman correlation with the instance index plus first-half / second-half means."""
    y = np.asarray(y, float)
    ok = ~np.isnan(y)
    if ok.sum() < 4:
        return "n/a"
    idx = np.arange(len(y))[ok]; v = y[ok]
    r1 = np.argsort(np.argsort(idx)); r2 = np.argsort(np.argsort(v))
    rho = np.corrcoef(r1, r2)[0, 1]
    h = len(v) // 2
    return f"first-half {v[:h].mean():.3f}  second-half {v[h:].mean():.3f}  spearman(idx) {rho:+.2f}"


def cmd_monitor(args) -> int:
    rows = list(csv.DictReader(open(args.csv)))
    if args.fold is not None:
        rows = [r for r in rows if str(r.get("fold")) == str(args.fold)]
    if not rows:
        print("no rows")
        return 1
    print(f"== {Path(args.csv).name}: {len(rows)} instance rows (fold={args.fold}) ==")
    col = lambda k: np.array([_f(r.get(k, "")) for r in rows])
    steps_rem = col("steps_rem"); steps_add = col("steps_add")
    d_rem = np.diff(np.concatenate([[steps_rem[0]], steps_rem])); d_add = np.diff(np.concatenate([[steps_add[0]], steps_add]))
    print(f"training steps per instance: remove mean {np.nanmean(d_rem[1:]) if len(d_rem)>1 else d_rem[0]:.1f}, add mean {np.nanmean(d_add[1:]) if len(d_add)>1 else d_add[0]:.1f}; "
          f"cumulative remove {steps_rem[-1]:.0f}, add {steps_add[-1]:.0f}; buffers rem {col('buf_rem')[-1]:.0f} add {col('buf_add')[-1]:.0f}")
    print(f"oracle calls per instance: {_trend(col('oracle_calls'))}")
    print(f"final size:                {_trend(col('final_size'))}")
    for short, label in (("rem", "remove"), ("add", "add")):
        n = col(f"{short}_n")
        if np.nansum(n) == 0:
            print(f"\n[{label}] no observations")
            continue
        print(f"\n[{label}] observations per instance mean {np.nanmean(n):.0f}, positive rate mean {np.nanmean(col(f'{short}_pos_rate')):.3f}")
        print(f"  skill vs base rate (>0 = model helps): {_trend(col(f'{short}_skill'))}")
        print(f"  AUC (0.5 = random):                    {_trend(col(f'{short}_auc'))}")
        print(f"  log-loss model / base:                 {np.nanmean(col(f'{short}_logloss')):.3f} / {np.nanmean(col(f'{short}_logloss_base')):.3f}")
        print(f"  tries to success (lower = better):     {_trend(col(f'{short}_tries_mean'))}")
        print(f"  rank advantage vs uniform (>0 good):   {_trend(col(f'{short}_rank_adv'))}")
        print(f"  BCE ema:                               {_trend(col(f'{short}_bce_ema'))}")
        print(f"  listwise ema:                          {_trend(col(f'{short}_listwise_ema'))}")
        print(f"  grad norm ema:                         {_trend(col(f'{short}_grad_norm_ema'))}")
        print(f"  update norm ema:                       {_trend(col(f'{short}_update_norm_ema'))}")
        print(f"  head disagreement (std):               {_trend(col(f'{short}_head_std'))}")
        dorm = col(f"{short}_dormant")
        if not np.all(np.isnan(dorm)):
            print(f"  dormant units frac:                    {_trend(dorm)}")
    if args.per_instance:
        print("\nidx  id    init final   OC  rem_n rem_skill rem_auc rem_tries rem_rankadv  add_n add_skill add_auc add_tries")
        for r in rows:
            print(f"{r['instance_idx']:>3s} {r['instance_id']:>5s} {r['initial_size']:>5s} {r['final_size']:>5s} {r['oracle_calls']:>5s} "
                  f"{r['rem_n']:>6s} {r['rem_skill']:>9s} {r['rem_auc']:>7s} {r['rem_tries_mean']:>9s} {r['rem_rank_adv']:>11s}  "
                  f"{r['add_n']:>5s} {r['add_skill']:>9s} {r['add_auc']:>7s} {r['add_tries_mean']:>9s}")
    return 0


# --------------------------------------------------------------------------- feedback

def cmd_feedback(args) -> int:
    """Calibration feedback: per-instance table across scopes plus, for v2-family monitor CSVs,
    the per-phase breakdown (oracle calls and successes in remove / swap / add, accepted block sizes,
    retries, prequential skill/AUC/tries)."""
    scopes = dict(s.split("=", 1) for s in args.scopes)
    data = {n: load_fold(sc, args.fold) or {} for n, sc in scopes.items()}
    ids = sorted(set.intersection(*[set(d) for d in data.values() if d]) if any(data.values()) else set(), key=lambda x: int(x) if x.isdigit() else x)
    names = list(scopes)
    print(f"== Per instance (fold {args.fold}), GED / OC ==")
    print("id     " + "".join(f"{n:>16s}" for n in names))
    for i in ids:
        print(f"{i:>6s} " + "".join(f"{data[n][i]['GraphEditDistance']:7.0f}/{data[n][i]['OracleCalls']:<8.0f}" for n in names))
    print("mean   " + "".join(f"{np.mean([data[n][i]['GraphEditDistance'] for i in ids]):7.2f}/{np.mean([data[n][i]['OracleCalls'] for i in ids]):<8.0f}" for n in names))
    base = names[0]
    for n in names[1:]:
        g = [(data[n][i]['GraphEditDistance'], data[base][i]['GraphEditDistance']) for i in ids]
        print(f"  {n:8s} GED vs {base}: better {sum(a < b for a, b in g)} tie {sum(a == b for a, b in g)} worse {sum(a > b for a, b in g)}")
    for csv_path in args.monitor or []:
        rows = list(csv.DictReader(open(csv_path)))
        if args.fold is not None:
            rows = [r for r in rows if str(r.get("fold")) == str(args.fold)]
        if not rows:
            continue
        col = lambda k: np.array([_f(r.get(k, "")) for r in rows])
        print(f"\n== Phase breakdown: {Path(csv_path).name} ({len(rows)} instances) ==")
        tot = col("oracle_calls")
        for ph in ("remove", "swap", "add"):
            c = col(f"calls_{ph}"); sc = col(f"succ_{ph}")
            with np.errstate(all="ignore"):
                per = np.nansum(c) / max(1.0, np.nansum(sc))
            print(f"  {ph:6s}: calls {np.nansum(c):8.0f} ({100*np.nansum(c)/max(1,np.nansum(tot)):4.1f}% of total)  successes {np.nansum(sc):5.0f}  calls per success {per:8.1f}")
        print(f"  accepted removal block: mean {np.nanmean(col('accepted_block_mean')):.2f}  max {np.nanmax(col('accepted_block_max')):.0f}   retries per instance {np.nanmean(col('retries')):.1f}")
        for short, label in (("rem", "remove"), ("add", "add")):
            print(f"  [{label}] n/inst {np.nanmean(col(f'{short}_n')):.0f}  pos rate {np.nanmean(col(f'{short}_pos_rate')):.3f}  AUC {np.nanmean(col(f'{short}_auc')):.3f}  "
                  f"skill {np.nanmean(col(f'{short}_skill')):+.3f}  tries/success {np.nanmean(col(f'{short}_tries_mean')):.1f}  steps {col(f'steps_{short}')[-1]:.0f}  "
                  f"head std {np.nanmean(col(f'{short}_head_std')):.3f}")
        print("  per instance: id init final OC | calls rem/swap/add | succ rem/swap/add | block mean/max | retries | rem AUC | add AUC")
        for r in rows:
            print(f"    {r['instance_id']:>5s} {r['initial_size']:>4s} {r['final_size']:>4s} {r['oracle_calls']:>5s} | {r.get('calls_remove',''):>5s}/{r.get('calls_swap',''):>5s}/{r.get('calls_add',''):>5s} | "
                  f"{r.get('succ_remove',''):>3s}/{r.get('succ_swap',''):>3s}/{r.get('succ_add',''):>3s} | {r.get('accepted_block_mean',''):>5s}/{r.get('accepted_block_max',''):>3s} | {r.get('retries',''):>2s} | {r.get('rem_auc',''):>6s} | {r.get('add_auc',''):>6s}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare"); c.add_argument("scopes", nargs="+", help="name=scope, first is the reference")
    c.add_argument("--folds", type=int, nargs="+", default=[0]); c.add_argument("--per-id", action="store_true")
    c.set_defaults(fn=cmd_compare)
    l = sub.add_parser("logs"); l.add_argument("logs", nargs="+"); l.add_argument("--block", type=int, default=10)
    l.add_argument("--first", type=int, default=10); l.set_defaults(fn=cmd_logs)
    m = sub.add_parser("monitor"); m.add_argument("csv"); m.add_argument("--fold", default=None)
    m.add_argument("--per-instance", action="store_true"); m.set_defaults(fn=cmd_monitor)
    fb = sub.add_parser("feedback"); fb.add_argument("scopes", nargs="+"); fb.add_argument("--fold", type=int, default=0)
    fb.add_argument("--monitor", nargs="*", help="monitor CSVs of the v2-family scopes"); fb.set_defaults(fn=cmd_feedback)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
