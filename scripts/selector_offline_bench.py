#!/usr/bin/env python3
"""Offline benchmark of selector architectures on examples collected by a ``collect_examples`` run.

Replays the online protocol instance by instance: the model scores every example of the next instance
*before* seeing any of it (prequential, so every score is about a graph the model has never trained on),
then trains on that instance the way the online loop does (examples queued, a step every ``flush_every``
examples and on every flip, the checkpoint reload between instances, i.e. shrink-and-perturb). No oracle
calls, so a variant costs seconds to minutes instead of a queue slot.

Metrics per mode, over the instances after ``--warmup``:
  auc_within   mean over instances of the AUC of flips against non-flips inside the instance: can the
               model tell, in the graph at hand, which moves will work?
  lift20       share of the instance's flips that fall in the model's top 20 % of moves, divided by 0.2
               (1.0 = no better than random order)
  auc_pooled   AUC over all scored examples together
  spread       mean within-instance standard deviation of the move scores (logits)
  lift20_pol   lift20 of the order the online policy would actually try: Plackett-Luce sampling of
               score / T with Gumbel noise (T = 1, the configured temperature). If the scores barely
               spread, this order is close to random whatever the ranking.
  lift20_zT    the same with the scores standardised inside the candidate set, at temperature T

The labelled examples come from uniform proposals, so they are an unbiased sample of the moves the
search tries. Swap halves are used as the online loop uses them (removal halves feed the removal model).

    python scripts/selector_offline_bench.py --examples lab/output/selector_pretrain/proteins_v3_collect.pt \\
        --config lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v2e/generate_minimize0.jsonc
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.explainer.future.metaheuristic.Tagging.OnlineSelectorV2 import (  # noqa: E402
    MoveExample, OnlineNNEdgeSelectorV2)

P1 = {"shrink": 1.0, "perturb_std": 0.0, "l2_init": 0.0, "extra_replay_steps_remove": 3}
P2 = {"input_norm": "log_running"}
P3 = {"target": "flip_margin", "pos_weight": 3.0}
P4 = {"shared_trunk": True}
P5 = {"prior_decay": 200.0}
VARIANTS = OrderedDict([
    ("base (v2e)", {}),
    ("P1 learn", P1),
    ("P2 norm", P2),
    ("P1+P2", {**P1, **P2}),
    ("P1+P2+P3 flip+margin", {**P1, **P2, **P3}),
    ("P1+P2+P4 shared trunk", {**P1, **P2, **P4}),
    ("P1+P2+P5 prior decay", {**P1, **P2, **P5}),
    ("P1..P5", {**P1, **P2, **P3, **P4, **P5}),
    ("P7 GBM refit", "gbm"),
    ("P3 on base", P3),
    ("flip on base", {"target": "flip"}),
])


ZTEMPS = (1.0, 0.5, 0.25)       # temperatures of the standardised-score policy


def auc(scores, pos) -> float:
    scores, pos = np.asarray(scores, float), np.asarray(pos, bool)
    if pos.all() or (~pos).all():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    for v in np.unique(scores):
        m = scores == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n = pos.sum()
    return float((ranks[pos].sum() - n * (n + 1) / 2) / (n * (~pos).sum()))


def lift20(scores, pos) -> float:
    pos = np.asarray(pos, bool)
    if pos.sum() == 0 or len(pos) < 5:
        return float("nan")
    top = np.argsort(-np.asarray(scores))[:max(1, int(round(0.2 * len(pos))))]
    return float(pos[top].sum() / pos.sum() / 0.2)


def group(rows):
    inst = OrderedDict()
    for r in rows:
        inst.setdefault(r["instance"], []).append(r)
    return inst


def run_nn(rows, kw, seed, flush, p_keep, p_keep_swap, rng) -> dict:
    k = (rows[0]["feats"].shape[1] - 4 - OnlineNNEdgeSelectorV2.STRUCT_DIM) // 4
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        sel = OnlineNNEdgeSelectorV2(k=k, **kw)
    scored = {"remove": [], "add": []}
    for i, (iid, exs) in enumerate(group(rows).items()):
        # 1) prequential scores on the unseen instance
        with torch.no_grad():
            for mode in ("remove", "add"):
                net = sel.models[mode]; net.eval()
                for r in exs:
                    if r["mode"] == mode:
                        s = float(sel._pool(net(r["feats"].to(sel.device)), net).mean())
                        scored[mode].append((i, s, r["y"] >= 1.0))
        # 2) online training on it, as the search loop feeds the selector
        for r in exs:
            flip = r["y"] >= 1.0
            mode = r["mode"].split("_")[-1]
            if r["mode"] == "swap_add":
                continue                                  # v2e: swap_record_add false
            if r["mode"] == "swap_remove":
                if not (flip or rng.random() < p_keep_swap):
                    continue
                w = 1.0 if flip else 0.5
            else:
                if not (flip or r["size"] == 1 or rng.random() < p_keep):
                    continue
                w = 1.0 if r["size"] == 1 else 0.5
            sel.push_example(mode, MoveExample(feats=r["feats"], y=float(r["y"]),
                                               boot=torch.poisson(torch.ones(sel.n_heads)).float(), weight=w))
            if flip or len(sel.pending[mode]) >= flush[mode]:
                sel.train_step(mode)
        for mode in ("remove", "add"):
            sel.train_step(mode)                          # end-of-instance flush
        sel.shrink_and_perturb()                          # the checkpoint reload between instances
        sel.instances_seen += 1
    return scored


def run_gbm(rows, rng) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    scored = {"remove": [], "add": []}
    train = {"remove": ([], []), "add": ([], [])}
    models = {"remove": None, "add": None}
    for i, (iid, exs) in enumerate(group(rows).items()):
        for mode in ("remove", "add"):
            for r in exs:
                if r["mode"] == mode:
                    x = r["feats"].mean(0).numpy()[None]
                    s = float(models[mode].predict_proba(x)[0, 1]) if models[mode] is not None else rng.random()
                    scored[mode].append((i, s, r["y"] >= 1.0))
        for r in exs:
            mode = r["mode"].split("_")[-1]
            if r["mode"] == "swap_add":
                continue
            train[mode][0].append(r["feats"].mean(0).numpy()); train[mode][1].append(r["y"] >= 1.0)
        for mode in ("remove", "add"):                    # refit between instances
            X, y = train[mode]
            if len(y) >= 50 and 0 < sum(y) < len(y):
                models[mode] = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05,
                                                              random_state=0).fit(np.stack(X), np.array(y))
    return scored


def summarise(scored, warmup, temp: float = 1.0) -> dict:
    out = {}
    for mode, rows in scored.items():
        rows = [r for r in rows if r[0] >= warmup]
        if not rows:
            continue
        by = {}
        for i, s, p in rows:
            by.setdefault(i, ([], []))[0].append(s); by[i][1].append(p)
        aw = [auc(s, p) for s, p in by.values()]
        lf = [lift20(s, p) for s, p in by.values()]
        prng = np.random.default_rng(0)
        lp, lz = [], {t: [] for t in ZTEMPS}
        for s, p in by.values():                          # the order the sampling policy would try
            s = np.asarray(s, float)
            g = -np.log(-np.log(prng.random((20, len(s))) + 1e-12) + 1e-12)
            lp += [lift20(s / temp + gi, p) for gi in g]
            z = (s - s.mean()) / (s.std() + 1e-9)         # standardised within the candidate set
            for t in ZTEMPS:
                lz[t] += [lift20(z / t + gi, p) for gi in g]
        sp = [np.std(s) for s, _ in by.values() if len(s) > 1]
        out[mode] = {"n": len(rows), "flip_rate": float(np.mean([r[2] for r in rows])),
                     "auc_within": float(np.nanmean(aw)) if not np.all(np.isnan(aw)) else float("nan"),
                     "instances_scored": int(np.sum(~np.isnan(aw))),
                     "lift20": float(np.nanmean(lf)) if not np.all(np.isnan(lf)) else float("nan"),
                     "auc_pooled": auc([r[1] for r in rows], [r[2] for r in rows]),
                     "spread": float(np.mean(sp)) if sp else float("nan"),
                     "lift20_pol": float(np.nanmean(lp)) if not np.all(np.isnan(lp)) else float("nan"),
                     **{f"lift20_z{t}": (float(np.nanmean(v)) if not np.all(np.isnan(v)) else float("nan"))
                        for t, v in lz.items()}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--examples", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--variants", nargs="*", default=None, help="subset of variant names")
    ap.add_argument("--warmup", type=int, default=10); ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default=None); ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    rows = torch.load(args.examples, weights_only=False)
    conf = json.load(open(args.config))
    par = conf["doe-triplets"][0]["explainer"]["parameters"]["minimizer"]["parameters"]
    base_kw = dict(par.get("selector", {}))
    base_kw["device"] = torch.device("cpu")
    flush = {"remove": int(par.get("flush_every", 10)), "add": int(par.get("flush_every_add", par.get("flush_every", 10)))}
    p_keep, p_keep_swap = float(par.get("p_neg_keep", 0.3)), float(par.get("p_neg_keep_swap", 0.15))
    n_inst = len({r["instance"] for r in rows})
    print(f"{len(rows)} examples, {n_inst} instances, warmup {args.warmup}; modes: "
          + ", ".join(f"{m}={sum(r['mode'] == m for r in rows)}" for m in ("remove", "add", "swap_remove", "swap_add")))

    report = {}
    names = args.variants or list(VARIANTS)
    print(f"\n{'variant':24s}{'seed':>5s} | {'rem auc_w':>9s}{'lift20':>7s}{'lift_pol':>9s}{'z1':>6s}{'z.5':>6s}{'z.25':>6s}{'spread':>7s} | {'add auc_w':>9s}{'lift20':>7s}{'#inst':>6s} | {'time':>5s}")
    for name in names:
        spec = VARIANTS[name]
        for seed in args.seeds:
            t0 = time.time()
            rng = random.Random(seed)
            scored = run_gbm(rows, rng) if spec == "gbm" else run_nn(rows, {**base_kw, **spec, "seed": seed}, seed,
                                                                     flush, p_keep, p_keep_swap, rng)
            sm = summarise(scored, args.warmup, float(base_kw.get("temp_remove", 1.0)))
            report.setdefault(name, []).append({"seed": seed, **sm})
            r, a = sm.get("remove", {}), sm.get("add", {})
            nan = float("nan")
            print(f"{name:24s}{seed:5d} | {r.get('auc_within', nan):9.3f}{r.get('lift20', nan):7.2f}"
                  f"{r.get('lift20_pol', nan):9.2f}{r.get('lift20_z1.0', nan):6.2f}{r.get('lift20_z0.5', nan):6.2f}"
                  f"{r.get('lift20_z0.25', nan):6.2f}{r.get('spread', nan):7.2f} | {a.get('auc_within', nan):9.3f}"
                  f"{a.get('lift20', nan):7.2f}{a.get('instances_scored', 0):6d} | {time.time() - t0:4.0f}s", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
