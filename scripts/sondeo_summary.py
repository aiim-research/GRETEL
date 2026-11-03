#!/usr/bin/env python3
"""One row per cell (dataset x oracle) of the selector paper's sondeo: the three arms on fold 0 and the
measures the architecture analysis asks for (docs/selector/ANALISIS_CONTROLES.md, section 6).

Per cell, from the run logs, the per-instance dumps and the selector monitors:
  GED and minimizer calls of bls_v0, bls_v1 (random) and selector_v1 (sel), paired tests;
  (a) stuck share: instances selector_v1 never improved, and the GED they hold;
  (b) trials per removal success of the uniform arm (monitor rem_tries_mean) and of the selector;
  (c) GED split by move kind ((-) sweep, (-o) overshoot, (=) swap, (+) add) and calls by phase;
  (d) selector_v1 vs bls_v1 on the advancing instances only;
  (e) validation AUC of the fold-1 pretraining (models/pretrained/*.json).
Writes a JSON with everything per cell and per instance to --out.

    python scripts/sondeo_summary.py [--gen 1|2] [--cells proteins-sel tri-40-sel ...] [--out lab/output/sondeo_summary.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from analyze_selector_arms import parse, dumps  # noqa: E402
from result_paths import scope_root  # noqa: E402

from gen_selector_paper_matrix import CKPT_PREFIX, SEL1, UNT1, V1, cells  # noqa: E402

MAIN = ["proteins", "synthie", "bbbp", "aids", "bzr", "enzymes"]
TRI = ["tri-40", "tri-40n", "tri-80"]
MONITOR = REPO / "lab/output/selector_logs/selector_{tag}_{dataset}.csv"
KINDS = ["-", "-o", "=", "+"]
# Generation 1 (the first sondeo, 2026-10-31, margin 0.9): arm suffix, monitor tag (shared by every cell of
# a dataset), checkpoint prefix. Generation 2 (the configuration of ANALISIS_SONDEO.md section 5, margin 0.7):
# per-cell monitor tags and new names.
GEN = {1: {"rnd": ("sel-random", "selector-random"), "sel": ("sel", "selector"), "ckpt": "edge_selector_sel-pre", "margin": "-m90", "per_cell_tag": False},
       2: {"rnd": (V1.replace("dce-lcls-", ""), "v1"), "sel": (SEL1.replace("dce-lcls-", ""), "sel-v1"), "ckpt": CKPT_PREFIX, "margin": "-m70", "per_cell_tag": True}}


def all_cells(gen: int = 2) -> list[str]:
    if gen == 2:
        return [f"{c}-sel" for c in cells()]
    cells1 = [f"{ds}{o}-sel" for ds in MAIN for o in ("", "-m90", "-rob")]
    cells1 += [f"colors-3{o}-s20-sel" for o in ("", "-m90", "-rob")]
    cells1 += [f"{t}{s}-sel" for t in TRI for s in ("", "-gcn", "-gcn-m90", "-gcn-rob")]
    return cells1


def dataset_dir(scope: str) -> str:
    root = scope_root(scope)
    dirs = [p.name for p in root.iterdir() if p.is_dir()]
    assert len(dirs) == 1, (scope, dirs)
    return dirs[0]


def match_monitor(tag: str, dataset: str, init: dict[int, int], final: dict[int, float], kmin: dict[int, int], fold: int) -> dict[int, dict] | None:
    """The monitor CSV of a dataset holds every run that used it (every oracle, every fold, several of them
    appending at the same time on the server), so rows are picked per instance: same fold and id, same
    initial and final size, and the minimizer calls closest to the dump's (within 5 % or 20 calls)."""
    path = Path(str(MONITOR).format(tag=tag, dataset=dataset))
    if not path.exists():
        return None
    by = {}
    for r in csv.DictReader(open(path)):
        if int(r["fold"]) == fold:
            by.setdefault(int(r["instance_id"]), []).append(r)
    out = {}
    for k in init:
        cands = [r for r in by.get(k, []) if int(r["initial_size"]) == init[k] and int(r["final_size"]) == final[k]]
        if not cands:
            continue
        r = min(cands, key=lambda r: abs(float(r["oracle_calls"]) - kmin[k]))
        if abs(float(r["oracle_calls"]) - kmin[k]) <= max(20, 0.05 * kmin[k]):
            out[k] = r
    return out if len(out) >= 0.8 * len(init) else None


def fnum(r: dict, c: str) -> float:
    v = r.get(c, "")
    return float(v) if v not in ("", "nan", None) else np.nan


def pretrain(name: str, prefix: str) -> dict:
    p = REPO / f"models/pretrained/{prefix}_{name}-f1.json"
    if not p.exists():
        return {}
    d = json.load(open(p))
    r = d["modes"].get("remove", {})
    h = r.get("history", [])
    return {"examples": d["examples"], "instances": d["instances"], "rem_val": r.get("val"), "rem_base": r.get("base_rate"),
            "auc_last": (h[-1].get("auc_flip") if h else None), "auc_best": max((x.get("auc_flip") or 0) for x in h) if h else None,
            "skill_last": (h[-1].get("skill") if h else None)}


def pval(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(wilcoxon(x, y).pvalue) if (x != y).any() else 1.0


def cell_summary(prefix: str, fold: int = 0, gen: int = 2) -> dict:
    g = GEN[gen]
    arms = {"v0": f"{prefix}_dce_lcls-v0", "rnd": f"{prefix}_dce_lcls-{g['rnd'][0]}", "sel": f"{prefix}_dce_lcls-{g['sel'][0]}"}
    T, D = {}, {}
    for a, s in arms.items():
        try:
            T[a] = parse(s, fold); D[a] = dumps(s, fold)
        except FileNotFoundError:
            T[a] = {}; D[a] = {}
    present = [a for a in arms if T[a] and D[a]]
    done = sorted(k for k in D.get("rnd", {}) if all(k in D[a] and k in T[a] and T[a][k]["done"] for a in present))
    keys = [k for k in done if all(T[a][k]["init"] is not None for a in present)]
    out = {"cell": prefix, "arms_present": present, "n": len(keys), "no_initial": len(done) - len(keys)}
    if not keys or "sel" not in present or "rnd" not in present:
        return out
    gen = {k: next(T[a][k]["gen"] for a in present if T[a][k]["gen"] is not None) for k in keys}
    init = {k: T["rnd"][k]["init"] for k in keys}
    final = {a: {k: D[a][k][0] for k in keys} for a in present}
    kmin = {a: {k: D[a][k][1] - gen[k] for k in keys} for a in present}
    edges = {k: D["rnd"][k][3] for k in keys}
    for a in present:
        bad = [k for k in keys if T[a][k]["init"] != init[k]]
        if bad:
            print(f"  warning {prefix} {a}: {len(bad)} instances with a different initial size", file=sys.stderr)
    out["init_mean"] = float(np.mean([init[k] for k in keys]))
    out["edges_mean"] = float(np.mean([edges[k] for k in keys]))
    out["gen_calls_mean"] = float(np.mean([gen[k] for k in keys]))
    for a in present:
        f = np.array([final[a][k] for k in keys], float); c = np.array([kmin[a][k] for k in keys], float)
        last = np.array([T[a][k]["traj"][-1][0] if T[a][k]["traj"] else 0 for k in keys], float)
        out[f"{a}_ged"] = float(f.mean()); out[f"{a}_ged_med"] = float(np.median(f))
        out[f"{a}_calls"] = float(c.mean()); out[f"{a}_calls_med"] = float(np.median(c))
        out[f"{a}_after_last_share"] = float((c - last).sum() / max(1, c.sum()))
        out[f"{a}_improved"] = int(sum(final[a][k] < init[k] for k in keys))
        # GED by move kind
        agg = {kd: [0, 0] for kd in KINDS}
        for k in keys:
            prev = init[k]
            for e in T[a][k]["events"]:
                if e[0] == "F":
                    kd = e[1] if e[1] in agg else "-"
                    agg[kd][0] += 1; agg[kd][1] += prev - e[3]; prev = e[3]
        tot = sum(v[1] for v in agg.values()) or 1
        out[f"{a}_kind_ged_share"] = {kd: v[1] / tot for kd, v in agg.items()}
        out[f"{a}_kind_n"] = {kd: v[0] for kd, v in agg.items()}
        out[f"{a}_descent"] = float(sum(init[k] - final[a][k] for k in keys))
    # paired comparisons
    def cmp(a, b):
        x = [final[a][k] for k in keys]; y = [final[b][k] for k in keys]
        return {"better": int(sum(u < v for u, v in zip(x, y))), "tie": int(sum(u == v for u, v in zip(x, y))),
                "worse": int(sum(u > v for u, v in zip(x, y))), "p": pval(x, y),
                "calls_p": pval([kmin[a][k] for k in keys], [kmin[b][k] for k in keys])}
    if "v0" in present:
        out["rnd_vs_v0"] = cmp("rnd", "v0")
    out["sel_vs_rnd"] = cmp("sel", "rnd")
    # (a) stuck
    never = [k for k in keys if not T["sel"][k]["traj"]]
    never_rnd = [k for k in keys if not T["rnd"][k]["traj"]]
    fb = [k for k in keys if T["sel"][k]["traj"] and T["sel"][k]["events"][0][1] == "-o"]
    out["stuck_sel"] = len(never); out["stuck_rnd"] = len(never_rnd); out["fallback_rescued_sel"] = len(fb)
    out["stuck_ged_share_sel"] = float(sum(final["sel"][k] for k in never) / max(1, sum(final["sel"].values())))
    out["stuck_both"] = len([k for k in never if k in set(never_rnd)])
    out["stuck_v0_improved"] = int(sum(final["v0"][k] < init[k] for k in never)) if "v0" in present else None
    # (d) advancing instances
    adv = [k for k in keys if final["sel"][k] < init[k] or final["rnd"][k] < init[k]]
    if adv:
        x = [final["sel"][k] for k in adv]; y = [final["rnd"][k] for k in adv]
        out["adv"] = {"n": len(adv), "sel_ged": float(np.mean(x)), "rnd_ged": float(np.mean(y)), "p": pval(x, y),
                      "sel_calls": float(np.mean([kmin["sel"][k] for k in adv])), "rnd_calls": float(np.mean([kmin["rnd"][k] for k in adv]))}
    # (b) monitors
    ds = dataset_dir(arms["sel"])
    out["dataset_dir"] = ds
    cols = ["rem_n", "rem_auc", "rem_tries_mean", "rem_successes", "rem_pos_rate", "calls_remove", "calls_swap", "calls_add",
            "succ_remove", "succ_swap", "succ_add", "wall_s", "accepted_block_mean", "accepted_block_max"]
    for a in ("sel", "rnd"):
        tag = g[a][1] + (f"-{prefix[:-4]}" if g["per_cell_tag"] else "")
        rows = match_monitor(tag, ds, init, final[a], kmin[a], fold)
        if rows is None:
            out[f"{a}_mon"] = None; continue
        m = {c: float(np.nanmean([fnum(rows[k], c) for k in keys if k in rows])) for c in cols}
        oc = [json.loads(rows[k]["overshoot"])["calls"] for k in keys if k in rows and rows[k].get("overshoot")]
        os_ = sum((json.loads(rows[k]["overshoot"])["successes"] for k in keys if k in rows and rows[k].get("overshoot")), [])
        m["overshoot_calls"] = float(np.mean(oc)) if oc else np.nan; m["overshoot_successes"] = len(os_)
        m["matched"] = sum(1 for k in keys if k in rows)
        out[f"{a}_mon"] = m
    # (e)
    out["pretrain"] = pretrain(prefix[:-4], g["ckpt"])
    out["instances"] = {str(k): {"init": init[k], "edges": edges[k], **{f"{a}_ged": final[a][k] for a in present},
                                 **{f"{a}_calls": kmin[a][k] for a in present}} for k in keys}
    return out


def fmt(v, w=6, d=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return " " * (w - 1) + "-"
    return f"{v:{w}.{d}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gen", type=int, choices=(1, 2), default=2, help="1: the first sondeo's arms and margin 0.9; 2: the current ones")
    ap.add_argument("--cells", nargs="+", default=None)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", default="lab/output/sondeo_summary.json")
    args = ap.parse_args()
    rows = []
    for c in args.cells or all_cells(args.gen):
        rows.append(cell_summary(c, args.fold, args.gen))
    hdr = (f"{'cell':22s} {'n':>4s} {'init':>6s} {'v0':>6s} {'v1':>6s} {'sel':>6s} | {'v1<v0':>7s} {'p':>7s} | {'sel<v1':>7s} {'p':>7s} |"
           f" {'c_v0':>6s} {'c_v1':>6s} {'c_sel':>6s} | {'stuck':>5s} {'gedsh':>5s} {'fb':>3s} | {'tries1':>6s} {'triesS':>6s} {'pqAUC':>5s} |"
           f" {'swp':>4s} {'ovs':>4s} {'swa':>4s} {'add':>4s} | {'advS':>6s} {'adv1':>6s} {'p':>7s} | {'preAUC':>6s} {'base':>5s}")
    print(hdr)
    for r in rows:
        if r.get("n", 0) == 0 or "sel_ged" not in r:
            print(f"{r['cell']:22s} {r.get('n', 0):>4d}  (missing arms: {[a for a in ('v0', 'rnd', 'sel') if a not in r.get('arms_present', [])]})")
            continue
        rv = r.get("rnd_vs_v0") or {}; sv = r["sel_vs_rnd"]; sm = r.get("sel_mon") or {}; rm = r.get("rnd_mon") or {}
        ks = r["sel_kind_ged_share"]; adv = r.get("adv") or {}; pt = r.get("pretrain") or {}
        print(f"{r['cell']:22s} {r['n']:4d} {fmt(r['init_mean'])} {fmt(r.get('v0_ged'))} {fmt(r['rnd_ged'])} {fmt(r['sel_ged'])} |"
              f" {rv.get('better', 0):3d}/{rv.get('worse', 0):<3d} {fmt(rv.get('p'), 7, 3)} | {sv['better']:3d}/{sv['worse']:<3d} {fmt(sv['p'], 7, 3)} |"
              f" {fmt(r.get('v0_calls'), 6, 0)} {fmt(r['rnd_calls'], 6, 0)} {fmt(r['sel_calls'], 6, 0)} |"
              f" {r['stuck_sel']:5d} {fmt(r['stuck_ged_share_sel'] * 100, 5, 0)} {r['fallback_rescued_sel']:3d} |"
              f" {fmt(rm.get('rem_tries_mean'), 6, 2)} {fmt(sm.get('rem_tries_mean'), 6, 2)} {fmt(sm.get('rem_auc'), 5, 2)} |"
              f" {fmt(ks['-'] * 100, 4, 0)} {fmt(ks['-o'] * 100, 4, 0)} {fmt(ks['='] * 100, 4, 0)} {fmt(ks['+'] * 100, 4, 0)} |"
              f" {fmt(adv.get('sel_ged'))} {fmt(adv.get('rnd_ged'))} {fmt(adv.get('p'), 7, 3)} | {fmt(pt.get('auc_last'), 6, 2)} {fmt(pt.get('rem_base'), 5, 2)}")
    json.dump(rows, open(args.out, "w"), indent=1)
    print(f"\nwritten {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
