#!/usr/bin/env python3
"""Analyse selector runs: per-instance comparison across scopes, start-of-fold learning curves from
the run logs, and within-fold training efficiency from the prequential monitor CSV.

Examples (repo root)::

    python scripts/analyze_selector_runs.py compare \
        base=synthie_dce_lcls v1=synthie_dce_lcls-net v2=synthie_dce_lcls-net-v2 --folds 0 1

    python scripts/analyze_selector_runs.py logs synthie_dce_lcls-net-v2:0 synthie_dce_lcls-net-v2:1 --block 10

    python scripts/analyze_selector_runs.py monitor lab/output/selector_logs/selector_v2_Synthie-<hash>.csv

    python scripts/analyze_selector_runs.py stats base=proteins40_dce_lcls v1=proteins40_dce_lcls-net \
        v2e=proteins40_dce_lcls-net-v2e v2e-r=proteins40_dce_lcls-net-v2e-random --ref base v1 v2e-r

    python scripts/analyze_selector_runs.py training --variant v2g --prefix proteins40   # or v1
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
sys.path.insert(0, str(REPO / "scripts"))
from result_paths import monitor_files, open_log, scope_root  # noqa: E402

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
    root = scope_root(scope)
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
    for line in open_log(path):
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


# ----------------------------------------------------------------------------- stats

def _holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def _paired(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    """Paired comparison of y (candidate) against x (reference); lower is better for both metrics."""
    from scipy.stats import wilcoxon
    d = y - x
    nz = d[np.abs(d) > 1e-9]
    out = {"n": len(d), "better": int((d < -1e-9).sum()), "tie": int((np.abs(d) <= 1e-9).sum()),
           "worse": int((d > 1e-9).sum()), "mean_x": float(x.mean()), "mean_y": float(y.mean()),
           "mean_d": float(d.mean()), "median_d": float(np.median(d))}
    boot = rng.choice(d, size=(5000, len(d)), replace=True).mean(axis=1)
    out["ci_lo"], out["ci_hi"] = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    if len(nz) == 0:
        out["p"], out["r"] = 1.0, 0.0
        return out
    out["p"] = float(wilcoxon(nz, alternative="two-sided").pvalue)
    # matched-pairs rank-biserial correlation, sign flipped so that r > 0 means the candidate is lower
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1.0
    # average ranks over ties in |d|
    absd = np.abs(nz)
    for v in np.unique(absd):
        m = absd == v
        ranks[m] = ranks[m].mean()
    wpos, wneg = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    out["r"] = float((wneg - wpos) / (wpos + wneg))
    return out


def cmd_stats(args) -> int:
    from scipy.stats import friedmanchisquare
    scopes = dict(s.split("=", 1) for s in args.scopes)
    data = {}
    for name, scope in scopes.items():
        r = load_fold(scope, args.fold, args.run)
        if r is None:
            print(f"  (missing) {scope} fold {args.fold}", file=sys.stderr)
            continue
        data[name] = r
    names = [n for n in scopes if n in data]
    common = sorted(set.intersection(*(set(data[n]) for n in names)), key=lambda k: int(k) if k.isdigit() else k)
    print(f"\n{len(common)} instances common to {len(names)} methods (fold {args.fold}, run {args.run})")
    mets = {"GED": "GraphEditDistance", "OC": "OracleCalls"}
    print(f"\n{'method':12s}" + "".join(f"{m+' mean':>12s}{m+' med':>10s}" for m in mets))
    for n in names:
        row = f"{n:12s}"
        for met in mets.values():
            v = np.array([data[n][k][met] for k in common])
            row += f"{np.nanmean(v):12.2f}{np.nanmedian(v):10.1f}"
        print(row)
    if len(names) >= 3:
        print("\nFriedman over all methods:")
        for m, met in mets.items():
            cols = [np.array([data[n][k][met] for k in common]) for n in names]
            st = friedmanchisquare(*cols)
            print(f"  {m:4s} chi2 = {st.statistic:8.2f}  p = {st.pvalue:.2e}")
    refs = args.ref or names[:1]
    rng = np.random.default_rng(0)
    tests = []
    for ref in refs:
        for n in names:
            if n == ref or n in refs[:refs.index(ref)]:
                continue
            for m, met in mets.items():
                x = np.array([data[ref][k][met] for k in common]); y = np.array([data[n][k][met] for k in common])
                tests.append((n, ref, m, _paired(x, y, rng)))
    adj = _holm([t[3]["p"] for t in tests])
    print(f"\nPaired two-sided Wilcoxon signed-rank, candidate vs reference (Holm over {len(tests)} tests).")
    print("d = candidate - reference (negative = candidate better); r = rank-biserial, > 0 favours candidate.")
    print(f"{'candidate':12s}{'reference':12s}{'met':>5s}{'W/T/L':>12s}{'mean d':>10s}{'95% CI':>24s}{'med d':>8s}{'r':>7s}{'p':>10s}{'p_holm':>10s}")
    for (n, ref, m, o), pa in zip(tests, adj):
        wtl = f"{o['better']}/{o['tie']}/{o['worse']}"
        ci = f"[{o['ci_lo']:.1f}, {o['ci_hi']:.1f}]"
        flag = " *" if pa < args.alpha else ""
        print(f"{n:12s}{ref:12s}{m:>5s}{wtl:>12s}{o['mean_d']:10.2f}{ci:>24s}{o['median_d']:8.1f}{o['r']:7.2f}{o['p']:10.2e}{pa:10.2e}{flag}")
    if args.per_id:
        print("\nid    " + "".join(f"{n:>14s}" for n in names) + "   (GED/OC)")
        for k in common:
            print(f"{k:>5s} " + "".join(f"{data[n][k]['GraphEditDistance']:>8.0f}/{data[n][k]['OracleCalls']:<5.0f}" for n in names))
    return 0


# -------------------------------------------------------------------------- training



def _read_monitor(tag: str, dataset: str) -> list[dict]:
    files = monitor_files(tag, dataset)
    if not files:
        return []
    return list(csv.DictReader(open(files[0])))


def _f(row: dict, key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def _arms(variant: str, prefix: str) -> dict[str, tuple[str, str]]:
    """arm -> (results scope, monitor tag)."""
    if variant == "v1":
        return {"learned": (f"{prefix}_dce_lcls-net", "v1"), "frozen": (f"{prefix}_dce_lcls-net-frozen", "v1-frozen")}
    arms = {"learned": (f"{prefix}_dce_lcls-net-{variant}", variant),
            "frozen": (f"{prefix}_dce_lcls-net-{variant}-frozen", f"{variant}-frozen")}
    if variant != "v2":
        arms["random"] = (f"{prefix}_dce_lcls-net-{variant}-random", f"{variant}-random")
    return arms


def cmd_training(args) -> int:
    from scipy.stats import spearmanr
    arms = _arms(args.variant, args.prefix)
    res = {a: load_fold(sc, args.fold, args.run) for a, (sc, _) in arms.items()}
    mon = {a: {r["instance_id"]: r for r in _read_monitor(tag, args.dataset)} for a, (_, tag) in arms.items()}
    for a, (sc, tag) in arms.items():
        print(f"  {a:8s} results {'ok' if res[a] else 'MISSING'} ({sc})   monitor rows {len(mon[a])} (tag {tag})")

    # ---- Q1: how much does the learned model train?
    rows = _read_monitor(arms["learned"][1], args.dataset)
    if rows:
        print(f"\nQ1. Training volume of the learned arm over {len(rows)} instances (cumulative)")
        print(f"{'inst':>5s}{'steps rem':>11s}{'steps add':>11s}{'ex rem':>9s}{'ex add':>9s}{'drift rem':>11s}{'drift add':>11s}"
              f"{'skill rem':>11s}{'auc rem':>9s}{'calls':>7s}")
        marks = sorted({1, 2, 5, 10, 20, 30, len(rows)} & set(range(1, len(rows) + 1)))
        for i in marks:
            r = rows[i - 1]
            print(f"{i:5d}{_f(r,'steps_rem'):11.0f}{_f(r,'steps_add'):11.0f}{_f(r,'examples_rem'):9.0f}{_f(r,'examples_add'):9.0f}"
                  f"{_f(r,'rem_drift'):11.3f}{_f(r,'add_drift'):11.3f}{_f(r,'rem_skill'):11.3f}{_f(r,'rem_auc'):9.3f}"
                  f"{_f(r,'oracle_calls'):7.0f}")
        half = len(rows) // 2
        for lab, part in (("first half", rows[:half]), ("second half", rows[half:])):
            sk = [_f(r, "rem_skill") for r in part]; ll = [_f(r, "rem_logloss") for r in part]
            if np.all(np.isnan(sk)):
                print(f"  prequential removal skill, {lab}: not recorded for this selector")
                continue
            print(f"  prequential removal skill, {lab}: mean {np.nanmean(sk):+.3f} (log-loss {np.nanmean(ll):.3f})")
        fz = list(mon.get("frozen", {}).values())
        if fz:
            print(f"  same metric for the frozen arm: skill {np.nanmean([_f(r,'rem_skill') for r in fz]):+.3f} "
                  f"(what the untrained network already knows)")

    # ---- Q2: does the training improve the search?
    def per_inst(arm: str) -> dict[str, dict[str, float]]:
        out = {}
        for iid, m in (res[arm] or {}).items():
            row = {"GED": m.get("GraphEditDistance", float("nan")), "OC": m.get("OracleCalls", float("nan"))}
            r = mon[arm].get(iid)
            if r:
                sr = _f(r, "succ_remove")
                row["calls/rem-succ"] = _f(r, "calls_remove") / sr if sr > 0 else float("nan")
                row["rem succ"] = sr
                row["order"] = _f(r, "instance_idx")
            out[iid] = row
        return out

    data = {a: per_inst(a) for a in arms if res[a]}
    pairs = [("learned", "frozen", "effect of training"), ("frozen", "random", "effect of an untrained ranking"),
             ("learned", "random", "total effect of the model")]
    rng = np.random.default_rng(0)
    print("\nQ2. Per-instance paired comparison (d = first - second, negative = first better; "
          "'rem succ' is higher = better, so its sign reads the other way)")
    print(f"{'pair':22s}{'metric':>15s}{'n':>4s}{'W/T/L':>10s}{'mean d':>10s}{'95% CI':>22s}{'r':>7s}{'p':>10s}")
    for a, b, what in pairs:
        if a not in data or b not in data:
            continue
        common = sorted(set(data[a]) & set(data[b]), key=lambda k: int(k) if k.isdigit() else k)
        for met in ("GED", "OC", "calls/rem-succ", "rem succ"):
            x = np.array([data[b][k].get(met, np.nan) for k in common]); y = np.array([data[a][k].get(met, np.nan) for k in common])
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() < 3:
                continue
            o = _paired(x[ok], y[ok], rng)
            print(f"{a+'-'+b:22s}{met:>15s}{o['n']:4d}{o['better']:>4d}/{o['tie']}/{o['worse']:<3d}{o['mean_d']:10.2f}"
                  f"{'[%.1f, %.1f]' % (o['ci_lo'], o['ci_hi']):>22s}{o['r']:7.2f}{o['p']:10.2e}")
        print(f"{'':22s}  ({what})")
    # does the training effect grow with experience? learned - frozen against processing order
    if "learned" in data and "frozen" in data:
        common = [k for k in data["learned"] if k in data["frozen"] and "order" in data["learned"][k]]
        if len(common) >= 6:
            order = np.array([data["learned"][k]["order"] for k in common])
            print("\nTrend of the training effect with experience (Spearman of learned - frozen against processing order;"
                  " rho < 0 = the learned arm pulls ahead as it trains)")
            for met in ("GED", "OC", "calls/rem-succ"):
                d = np.array([data["learned"][k].get(met, np.nan) - data["frozen"][k].get(met, np.nan) for k in common])
                ok = ~np.isnan(d)
                if ok.sum() >= 6:
                    rho, p = spearmanr(order[ok], d[ok])
                    first = d[ok][order[ok] < np.median(order[ok])]; last = d[ok][order[ok] >= np.median(order[ok])]
                    print(f"  {met:15s} rho = {rho:+.2f} (p = {p:.2f})   mean d first half {np.mean(first):+.2f}, second half {np.mean(last):+.2f}")
    return 0


# --------------------------------------------------------------------------- seeds

def cmd_seeds(args) -> int:
    """Pool several minimizer seeds. Seed 0 is the base scope, seed s > 0 the scope with ``-s<s>``.

    Primary test: per instance, average each method's metric over the seeds, then a paired two-sided
    Wilcoxon on the 40 instance means (the seeds of one instance are not independent, so the instance
    is the unit). Also the per-seed tests, to see whether the sign holds in every seed.
    """
    scopes = dict(s.split("=", 1) for s in args.scopes)
    data = {}
    for name, base in scopes.items():
        per_seed = {}
        for sd in args.seeds:
            r = load_fold(base if sd == 0 else f"{base}-s{sd}", args.fold, args.run)
            if r is None:
                print(f"  (missing) {name} seed {sd}", file=sys.stderr)
                continue
            per_seed[sd] = r
        data[name] = per_seed
    names = [n for n in scopes if data[n]]
    seeds = [sd for sd in args.seeds if all(sd in data[n] for n in names)]
    common = sorted(set.intersection(*(set(data[n][sd]) for n in names for sd in seeds)),
                    key=lambda k: int(k) if k.isdigit() else k)
    print(f"\n{len(common)} instances x seeds {seeds} for {len(names)} methods")
    mets = {"GED": "GraphEditDistance", "OC": "OracleCalls"}
    mean = {n: {m: np.array([np.mean([data[n][sd][k][met] for sd in seeds]) for k in common])
                for m, met in mets.items()} for n in names}
    print(f"\n{'method':12s}" + "".join(f"{m+' mean':>11s}{m+' med':>9s}" for m in mets) + "   (seed-averaged per instance)")
    for n in names:
        print(f"{n:12s}" + "".join(f"{mean[n][m].mean():11.2f}{np.median(mean[n][m]):9.1f}" for m in mets))
    refs = args.ref or names[:1]
    rng = np.random.default_rng(0)
    tests = []
    for ref in refs:
        for n in names:
            if n == ref or n in refs[:refs.index(ref)]:
                continue
            for m in mets:
                tests.append((n, ref, m, _paired(mean[ref][m], mean[n][m], rng)))
    adj = _holm([t[3]["p"] for t in tests])
    print(f"\nPaired two-sided Wilcoxon on seed-averaged instance values (Holm over {len(tests)} tests); d = candidate - reference")
    print(f"{'candidate':12s}{'reference':12s}{'met':>5s}{'W/T/L':>12s}{'mean d':>10s}{'95% CI':>24s}{'r':>7s}{'p':>10s}{'p_holm':>10s}   per-seed mean d")
    for (n, ref, m, o), pa in zip(tests, adj):
        per = []
        for sd in seeds:
            x = np.array([data[ref][sd][k][mets[m]] for k in common]); y = np.array([data[n][sd][k][mets[m]] for k in common])
            per.append(f"{(y - x).mean():+.1f}")
        flag = " *" if pa < args.alpha else ""
        print(f"{n:12s}{ref:12s}{m:>5s}{o['better']:>5d}/{o['tie']}/{o['worse']:<4d}{o['mean_d']:10.2f}"
              f"{'[%.1f, %.1f]' % (o['ci_lo'], o['ci_hi']):>24s}{o['r']:7.2f}{o['p']:10.2e}{pa:10.2e}{flag}   " + " ".join(per))
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
    st = sub.add_parser("stats", help="paired per-instance significance tests between scopes")
    st.add_argument("scopes", nargs="+", help="name=scope")
    st.add_argument("--ref", nargs="*", help="reference names (default: the first scope)")
    st.add_argument("--fold", type=int, default=0); st.add_argument("--run", type=int, default=1)
    st.add_argument("--alpha", type=float, default=0.05); st.add_argument("--per-id", action="store_true")
    st.set_defaults(fn=cmd_stats)
    sd = sub.add_parser("seeds", help="pool minimizer seeds (seed s > 0 = scope with -s<s>) and test on instance means")
    sd.add_argument("scopes", nargs="+", help="name=base scope"); sd.add_argument("--ref", nargs="*")
    sd.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); sd.add_argument("--fold", type=int, default=0)
    sd.add_argument("--run", type=int, default=1); sd.add_argument("--alpha", type=float, default=0.05)
    sd.set_defaults(fn=cmd_seeds)
    tr = sub.add_parser("training", help="how much a selector trains, and what the training buys (learned/frozen/random)")
    tr.add_argument("--variant", required=True, help="v1, v2, v2b ... v2g"); tr.add_argument("--prefix", default="proteins40")
    tr.add_argument("--dataset", default="PROTEINS"); tr.add_argument("--fold", type=int, default=0)
    tr.add_argument("--run", type=int, default=1); tr.set_defaults(fn=cmd_training)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
