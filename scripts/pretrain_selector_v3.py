#!/usr/bin/env python3
"""Pretrain the v2/v3 edge selector offline on examples collected by a ``collect_examples`` run.

The examples come from instances disjoint from the evaluation set. They are split by instance into
train and validation, so the validation scores measure what the ranker learns that transfers to graphs
it has never seen. That is the question the online runs cannot answer cleanly, because there every
prediction is about the instance the model is currently searching.

For each mode (remove, add) it trains with the selector's own BCE loss, early-stops on validation
log-loss, and reports validation log-loss against a base-rate predictor (skill > 0 means the model beats
"always predict the average") and AUC of flips against non-flips. The checkpoint it writes starts from
the same seeded initialisation as a cold learned run, uses the pretrained weights as the L2-init anchor
and drift reference, and carries a reservoir of the training examples as its replay buffer.

    python scripts/pretrain_selector_v3.py \\
        --examples lab/output/selector_pretrain/proteins_v3_collect.pt \\
        --config lab/config/generate_minimize/proteins40/dce/dce-lcls-net-v3/generate_minimize0.jsonc \\
        --out models/pretrained/edge_selector_v3-pre_PROTEINS.pt
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.explainer.future.metaheuristic.Tagging.OnlineSelectorV2 import (  # noqa: E402
    MoveExample, OnlineNNEdgeSelectorV2)


def auc(scores: np.ndarray, pos: np.ndarray) -> float:
    if pos.all() or (~pos).all():
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    for v in np.unique(scores):                      # average ranks over ties
        m = scores == v
        ranks[m] = ranks[m].mean()
    n_pos = pos.sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (~pos).sum()))


@torch.no_grad()
def evaluate(sel: OnlineNNEdgeSelectorV2, mode: str, exs: list[MoveExample], base_p: float) -> dict:
    net = sel.models[mode]
    net.eval()
    probs = []
    for ex in exs:
        logit = sel._pool(net(ex.feats.to(sel.device)), net).mean()
        probs.append(float(torch.sigmoid(logit)))
    p = np.clip(np.array(probs), 1e-6, 1 - 1e-6)
    y = np.array([ex.y for ex in exs])
    if sel.target != "soft":                         # score against the label the model is trained on
        y = (y >= 1.0).astype(float)
    ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    b = min(max(base_p, 1e-6), 1 - 1e-6)
    ll_base = float(-(y * math.log(b) + (1 - y) * math.log(1 - b)).mean())
    return {"n": len(exs), "flip_rate": float((y >= 1.0).mean()), "logloss": ll, "logloss_base": ll_base,
            "skill": 1 - ll / ll_base, "auc_flip": auc(p, y >= 1.0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--examples", required=True, nargs="+", help="one or more collected example files")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--report", default=None)
    ap.add_argument("--val-frac", type=float, default=0.25); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [r for f in args.examples for r in torch.load(f, weights_only=False)]
    conf = json.load(open(args.config))
    par = conf["doe-triplets"][0]["explainer"]["parameters"]["minimizer"]["parameters"]
    kw = dict(par.get("selector", {}))
    kw.setdefault("seed", par.get("seed"))
    if kw.get("evidence_features"):
        # v4: the network also reads the within-instance evidence block stored with each example
        rows = [r for r in rows if "evid" in r]
        for r in rows:
            r["feats"] = torch.cat([r["feats"], r["evid"].to(r["feats"].dtype)], dim=1)
    input_dim = rows[0]["feats"].shape[1]
    extra = OnlineNNEdgeSelectorV2.EVID_DIM if kw.get("evidence_features") else 0
    k = (input_dim - 4 - OnlineNNEdgeSelectorV2.STRUCT_DIM - extra) // 4
    with torch.random.fork_rng(devices=[]):          # same initialisation as a cold learned run
        torch.manual_seed(par.get("seed") if par.get("seed") is not None else 0)
        sel = OnlineNNEdgeSelectorV2(k=k, **kw)
    assert sel.input_dim == input_dim, (sel.input_dim, input_dim)

    rng = random.Random(args.seed)
    instances = sorted({r["instance"] for r in rows})
    rng.shuffle(instances)
    n_val = max(1, int(round(args.val_frac * len(instances))))
    val_inst = set(instances[:n_val])
    report = {"examples": len(rows), "instances": len(instances), "val_instances": sorted(val_inst), "modes": {}}
    print(f"{len(rows)} examples from {len(instances)} instances ({n_val} held out for validation), k={k}")

    for mode in ("remove", "add"):
        mk = lambda r: MoveExample(feats=r["feats"], y=float(r["y"]),
                                   boot=torch.poisson(torch.ones(sel.n_heads)).float(), weight=1.0)
        tr = [mk(r) for r in rows if r["mode"] == mode and r["instance"] not in val_inst]
        va = [mk(r) for r in rows if r["mode"] == mode and r["instance"] in val_inst]
        if len(tr) < 50 or len(va) < 20:
            print(f"[{mode}] too few examples (train {len(tr)}, val {len(va)}), left untrained")
            report["modes"][mode] = {"train": len(tr), "val": len(va), "skipped": True}
            continue
        base_p = float(np.mean([(e.y if sel.target == "soft" else float(e.y >= 1.0)) for e in tr]))
        hist = [{"epoch": 0, **evaluate(sel, mode, va, base_p)}]
        print(f"[{mode}] train {len(tr)}  val {len(va)}  base rate {base_p:.3f}")
        print(f"  epoch  0  val logloss {hist[0]['logloss']:.4f} (base {hist[0]['logloss_base']:.4f})  "
              f"skill {hist[0]['skill']:+.3f}  auc {hist[0]['auc_flip']:.3f}")
        best, best_state, bad = float("inf"), None, 0
        net, opt = sel.models[mode], sel.opts[mode]
        for ep in range(1, args.epochs + 1):
            net.train()
            rng.shuffle(tr)
            for i in range(0, len(tr), args.batch):
                opt.zero_grad(set_to_none=True)
                loss = sel._bce_loss(mode, tr[i:i + args.batch])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.trainable_parameters(), max_norm=1.0)
                opt.step()
                sel.train_steps[mode] += 1
            h = {"epoch": ep, **evaluate(sel, mode, va, base_p)}
            hist.append(h)
            print(f"  epoch {ep:2d}  val logloss {h['logloss']:.4f}  skill {h['skill']:+.3f}  auc {h['auc_flip']:.3f}")
            if h["logloss"] < best - 1e-4:
                best, bad = h["logloss"], 0
                best_state = {kk: v.detach().clone() for kk, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        final = evaluate(sel, mode, va, base_p)
        print(f"  best: val logloss {final['logloss']:.4f} (base {final['logloss_base']:.4f})  "
              f"skill {final['skill']:+.3f}  auc {final['auc_flip']:.3f}")
        report["modes"][mode] = {"train": len(tr), "val": len(va), "base_rate": base_p, "history": hist, "final": final}
        for ex in tr:                                   # warm replay buffer for the online phase
            sel.buffers[mode].add(ex)

    # the pretrained weights become the anchor of L2-init and the reference of the drift metric
    sel.init_params = {m: [p.detach().clone() for p in net.trainable_parameters()] for m, net in sel.models.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sel.save(args.out)
    rep = Path(args.report or Path(args.examples[0]).with_suffix(".report.json"))
    rep.write_text(json.dumps(report, indent=1))
    print(f"checkpoint {args.out}\nreport {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
