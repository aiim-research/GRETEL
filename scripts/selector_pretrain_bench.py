#!/usr/bin/env python3
"""Compare offline-pretrained removal rankers by cross-validation over instances.

The v3b-pre-frozen selector ranks with a network trained offline and never updated, so what matters for
v4 is which pretrained ranker orders the removal moves of unseen graphs best, and whether more
pretraining data would help. The collected examples are split by instance into ``--folds`` folds; each
model is trained on the other folds and scored on the held-out instances, never on a graph it saw.

Metrics on the held-out instances (removal moves, flips as positives):
  auc_w     mean within-instance AUC of flips against non-flips
  lift20    share of an instance's flips in the model's top 20 %, over 0.2 (greedy order)
  lift_z25  the same for the order the v3b sweep policy tries (standardised scores, T 0.25, Gumbel)

``--curve`` repeats the NN and GBM with fewer training instances per fold, to see whether the held-out
quality is still rising with data (if it is, collecting more instances is worth it).

    python scripts/selector_pretrain_bench.py --examples lab/output/selector_pretrain/proteins_v3_collect.pt \\
        --config lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v3b/generate_minimize0.jsonc --curve
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from src.explainer.future.metaheuristic.Tagging.OnlineSelectorV2 import (  # noqa: E402
    MoveExample, OnlineNNEdgeSelectorV2)
from selector_offline_bench import auc, lift20  # noqa: E402


def policy_lift(scores, pos, temp=0.25, draws=20, rng=None):
    s = np.asarray(scores, float)
    z = (s - s.mean()) / (s.std() + 1e-9)
    g = -np.log(-np.log(rng.random((draws, len(s))) + 1e-12) + 1e-12)
    return float(np.nanmean([lift20(z / temp + gi, pos) for gi in g]))


def metrics(by_inst, rng) -> dict:
    aw = [auc(s, p) for s, p in by_inst.values()]
    lf = [lift20(s, p) for s, p in by_inst.values()]
    lz = [policy_lift(s, p, rng=rng) for s, p in by_inst.values() if np.sum(p) > 0 and len(p) >= 5]
    m = lambda v: float(np.nanmean(v)) if not np.all(np.isnan(v)) else float("nan")
    return {"auc_w": m(aw), "lift20": m(lf), "lift_z25": m(lz), "instances": int(np.sum(~np.isnan(aw)))}


def train_nn(train_rows, kw, seed, epochs=30, batch=64, val_frac=0.15):
    k = (train_rows[0]["feats"].shape[1] - 4 - OnlineNNEdgeSelectorV2.STRUCT_DIM) // 4
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        sel = OnlineNNEdgeSelectorV2(k=k, **kw)
    rng = random.Random(seed)
    inst = sorted({r["instance"] for r in train_rows}); rng.shuffle(inst)
    val = set(inst[:max(1, int(round(val_frac * len(inst))))])
    mk = lambda r: MoveExample(feats=r["feats"], y=float(r["y"]), boot=torch.poisson(torch.ones(sel.n_heads)).float())
    tr = [mk(r) for r in train_rows if r["instance"] not in val]
    va = [mk(r) for r in train_rows if r["instance"] in val]
    net, opt = sel.models["remove"], sel.opts["remove"]
    for ex in tr:
        net.update_norm(ex.feats)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        net.train(); rng.shuffle(tr)
        for i in range(0, len(tr), batch):
            opt.zero_grad(set_to_none=True)
            loss = sel._bce_loss("remove", tr[i:i + batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.trainable_parameters(), 1.0)
            opt.step()
        with torch.no_grad():
            vl = float(sel._bce_loss("remove", va)) if va else 0.0
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {kk: v.detach().clone() for kk, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    @torch.no_grad()
    def score(r):
        return float(sel._pool(net(r["feats"]), net).mean())
    return score


def train_gbm(train_rows, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = np.stack([r["feats"].mean(0).numpy() for r in train_rows]); y = np.array([r["y"] >= 1.0 for r in train_rows])
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=True,
                                       validation_fraction=0.15, random_state=seed).fit(X, y)
    return lambda r: float(m.predict_proba(r["feats"].mean(0).numpy()[None])[0, 1])


def train_logreg(train_rows, seed):
    from sklearn.linear_model import LogisticRegression
    X = np.stack([r["feats"].mean(0).numpy() for r in train_rows]); y = np.array([r["y"] >= 1.0 for r in train_rows])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    m = LogisticRegression(max_iter=3000, C=1.0).fit((X - mu) / sd, y)
    return lambda r: float(m.decision_function(((r["feats"].mean(0).numpy() - mu) / sd)[None])[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--examples", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--curve", action="store_true"); ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--out", default=None); ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    rows = [r for r in torch.load(args.examples, weights_only=False) if r["mode"] in ("remove", "swap_remove")]
    par = json.load(open(args.config))["doe-triplets"][0]["explainer"]["parameters"]["minimizer"]["parameters"]
    base = {**par.get("selector", {}), "device": torch.device("cpu")}
    models = {
        "nn (v3b)": lambda tr, s: train_nn(tr, base, s),
        "nn wide 128x3": lambda tr, s: train_nn(tr, {**base, "hidden_dim": 128, "num_hidden_layers": 3}, s),
        "nn 1 head": lambda tr, s: train_nn(tr, {**base, "n_heads": 1, "prior_scale": 0.0}, s),
        "gbm": train_gbm,
        "logreg": train_logreg,
    }
    names = args.models or list(models)
    insts = sorted({r["instance"] for r in rows})
    random.Random(args.seed).shuffle(insts)
    folds = [set(insts[i::args.folds]) for i in range(args.folds)]
    sizes = [None] + ([9, 18, 36] if args.curve else [])
    print(f"{len(rows)} removal examples (single sweeps and swap halves) from {len(insts)} instances, {args.folds} folds")
    print(f"\n{'model':16s}{'train inst':>11s} | {'auc_w':>7s}{'lift20':>8s}{'lift_z25':>9s}{'#inst':>6s} | {'time':>5s}")
    report = {}
    for name in names:
        for size in sizes:
            if size is not None and name not in ("nn (v3b)", "gbm"):
                continue
            t0 = time.time()
            by_inst, rng = {}, np.random.default_rng(args.seed)
            n_train = []
            for f, held in enumerate(folds):
                pool = sorted({r["instance"] for r in rows} - held)
                if size is not None:
                    pool = random.Random(args.seed + f).sample(pool, min(size, len(pool)))
                pool = set(pool); n_train.append(len(pool))
                tr = [r for r in rows if r["instance"] in pool]
                score = models[name](tr, args.seed + f)
                for r in rows:
                    if r["instance"] in held and r["mode"] == "remove":
                        by_inst.setdefault(r["instance"], ([], []))[0].append(score(r))
                        by_inst[r["instance"]][1].append(r["y"] >= 1.0)
            m = metrics(by_inst, rng)
            report.setdefault(name, {})[str(size or "all")] = m
            print(f"{name:16s}{int(np.mean(n_train)):11d} | {m['auc_w']:7.3f}{m['lift20']:8.2f}{m['lift_z25']:9.2f}"
                  f"{m['instances']:6d} | {time.time() - t0:4.0f}s", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
