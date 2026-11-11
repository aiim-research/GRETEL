#!/usr/bin/env python3
"""Arm D of the selector paper (docs/v2/RUNS.md): what the traces of bls_v1-F, control-F and selector_v1-F say, on the
12 classic cells, folds 0 to 9.

Each traced run (``trace: true`` in ``local_search_selection_net_v2.py``) writes one JSON line per instance to
``trace_<model_tag>_<dataset>_f<fold>.jsonl``: the minimizer's oracle calls, how many edge edits and feature
substitutions were proposed for removal and for addition, and, after every accepted move, ``[calls so far, edge edits,
substitutions, feature cost]`` of the best solution (the feature cost is each substitution's L2 distance under the
identity correspondence, the same quantity as MatchedFeatureDistance when the seed is in the input's frame).

Writes ``trazas.md`` into OUT with three tables:

1. the mechanism: initial and final composition of the best solution per arm, the feature cost per substitution
   left, and the substitutions removed per substitution-removal proposed;
2. the decomposition of the selector's feature gain against bls_v1-F into fewer substitutions and cheaper ones;
3. convergence: mean E + F of the best solution after b minimizer calls, the median calls of the last improvement,
   and how often the selector reaches bls_v1-F's final cost earlier or later.

It also checks that the traced runs reproduce the untraced ones, instance by instance.

    python scripts/v2_traces.py OUT [--logs lab/output/selector_logs] [--results lab/output/results]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2_paper_tables import CELLS, load  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ARMS = [("bls_v1-F", "v2-fm-v1-{c}-t", "lcls-v1-f"), ("control-F", "v2-fm-sel-v1-untrained-{c}-t", "lcls-sel-v1-untrained-f"),
        ("selector_v1-F", "v2-fm-sel-v1-{c}-t", "lcls-sel-v1-f")]
BUDGETS = [25, 50, 100, 200, 400]


def read_traces(logs: Path, tag: str) -> dict:
    rows = {}
    for f in glob.glob(str(logs / f"trace_{tag}_*_f*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r.get("status") == "ok" and r.get("trace"):
                rows[(int(r["fold"]), str(r["instance_id"]))] = r
    return rows


def best_at(trace, b):
    v = None
    for k, e, _, f in trace:
        if k > b:
            break
        v = e + f
    return v


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--logs", default=str(REPO / "lab/output/selector_logs"))
    ap.add_argument("--results", default=str(REPO / "lab/output/results"))
    args = ap.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    logs, results = Path(args.logs), Path(args.results)
    T = {(c, a): read_traces(logs, tag.format(c=c)) for c in CELLS for a, tag, _ in ARMS}

    md = ["# Trazas del brazo D: bls_v1-F, control-F y selector_v1-F, 12 celdas clásicas, folds 0 a 9", "",
          "Generado por `scripts/v2_traces.py` (GRETEL). E = aristas editadas, Sub = sustituciones de features, "
          "F = coste de features de las sustituciones (distancia L2, correspondencia identidad). Llamadas: las del "
          "minimizador, sin las del generador.", ""]

    md += ["## 0. Las corridas trazadas reproducen las originales", "",
           "Instancias con S, F y llamadas idénticas entre la corrida trazada (`-t`) y la original.", "",
           "| Celda | " + " | ".join(a for a, _, _ in ARMS) + " |", "|---|" + "---|" * len(ARMS)]
    if any(results.glob("v2-*-sel_dcm_*-t")):
        for c in CELLS:
            row = []
            for a, _, scope in ARMS:
                o, t = load(results, f"v2-{c}-sel_dcm_{scope}"), load(results, f"v2-{c}-sel_dcm_{scope}-t")
                ks = set(o) & set(t)
                bad = sorted({k[0] for k in ks if any(abs(o[k].get(m, 0) - t[k].get(m, 0)) > 1e-6
                                                      for m in ("struct", "feat", "calls"))})
                same = sum(all(abs(o[k].get(m, 0) - t[k].get(m, 0)) <= 1e-6 for m in ("struct", "feat", "calls")) for k in ks)
                row.append(f"{same}/{len(ks)}" + (f" (difiere el fold {', '.join(map(str, bad))})" if bad else ""))
            md.append(f"| {c} | " + " | ".join(row) + " |")
    else:
        md.append("(sin resultados `-t` en --results)")

    md += ["", "## 1. Composición de la mejor solución", "",
           "Medias por instancia. F/Sub = coste de features por sustitución que queda al final. Rendimiento = "
           "sustituciones quitadas por cada retirada de sustitución propuesta.", "",
           "| Celda | Brazo | E inicial | Sub inicial | F inicial | E final | Sub final | E + Sub final | F final | F/Sub | "
           "rendimiento | llamadas |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in CELLS:
        for a, _, _ in ARMS:
            rows = list(T[(c, a)].values())
            if not rows:
                continue
            i = np.array([r["trace"][0] for r in rows], float); f = np.array([r["trace"][-1] for r in rows], float)
            prop = sum(r["proposed"]["remove"][1] for r in rows)
            yld = (i[:, 2] - f[:, 2]).sum() / prop if prop else float("nan")
            fps = f[:, 3].sum() / f[:, 2].sum() if f[:, 2].sum() else float("nan")
            md.append(f"| {c} | {a} | {i[:,1].mean():.2f} | {i[:,2].mean():.2f} | {i[:,3].mean():.2f} | {f[:,1].mean():.2f} | "
                      f"{f[:,2].mean():.2f} | {f[:,1:3].sum(1).mean():.2f} | {f[:,3].mean():.2f} | {fps:.2f} | {yld:.2f} | "
                      f"{np.mean([r['calls'] for r in rows]):.0f} |")

    md += ["", "## 2. De dónde sale la ganancia en F del selector frente a bls_v1-F", "",
           "ΔF = F(selector) − F(bls_v1-F), medias por instancia. Parte número = (Sub del selector − Sub de bls_v1-F) × "
           "F/Sub de bls_v1-F, lo que se gana por dejar menos sustituciones al mismo precio. El resto es el efecto precio. "
           "Las tres últimas columnas: fracción de instancias en que el selector deja menos, igual o más sustituciones.", "",
           "| Celda | ΔE | ΔSub | ΔF | parte número | parte precio | menos Sub | igual | más Sub |",
           "|---|---|---|---|---|---|---|---|---|"]
    for c in CELLS:
        a, b = T[(c, "selector_v1-F")], T[(c, "bls_v1-F")]
        ks = [k for k in a if k in b]
        if not ks:
            continue
        fa = np.array([a[k]["trace"][-1] for k in ks], float); fb = np.array([b[k]["trace"][-1] for k in ks], float)
        price = fb[:, 3].sum() / fb[:, 2].sum() if fb[:, 2].sum() else 0.0
        dsub = fa[:, 2].mean() - fb[:, 2].mean(); dF = fa[:, 3].mean() - fb[:, 3].mean(); d = fa[:, 2] - fb[:, 2]
        md.append(f"| {c} | {fa[:,1].mean() - fb[:,1].mean():+.2f} | {dsub:+.2f} | {dF:+.2f} | {dsub * price:+.2f} | "
                  f"{dF - dsub * price:+.2f} | {np.mean(d < 0):.2f} | {np.mean(d == 0):.2f} | {np.mean(d > 0):.2f} |")

    md += ["", "## 3. Convergencia", "",
           "E + F medio de la mejor solución tras b llamadas del minimizador, y mediana de las llamadas de la última mejora.", "",
           "| Celda | Brazo | " + " | ".join(f"b = {b}" for b in BUDGETS) + " | última mejora |",
           "|---|---|" + "---|" * (len(BUDGETS) + 1)]
    for c in CELLS:
        for a, _, _ in ARMS:
            rows = list(T[(c, a)].values())
            if not rows:
                continue
            M = np.array([[best_at(r["trace"], b) for b in BUDGETS] for r in rows], float)
            md.append(f"| {c} | {a} | " + " | ".join(f"{x:.1f}" for x in M.mean(0))
                      + f" | {np.median([r['trace'][-1][0] for r in rows]):.0f} |")
    md += ["", "Cuándo llega el selector al coste final E + F de bls_v1-F, instancia a instancia (fracciones). \"Nunca\" "
           "son las instancias en que el selector termina por encima.", "",
           "| Celda | antes | a la vez | después | nunca |", "|---|---|---|---|---|"]
    for c in CELLS:
        a, b = T[(c, "selector_v1-F")], T[(c, "bls_v1-F")]
        d = []
        for k in (k for k in b if k in a):
            target = b[k]["trace"][-1][1] + b[k]["trace"][-1][3] + 1e-9
            kb = next(r[0] for r in b[k]["trace"] if r[1] + r[3] <= target)
            ka = next((r[0] for r in a[k]["trace"] if r[1] + r[3] <= target), None)
            d.append(np.inf if ka is None else ka - kb)
        d = np.array(d)
        if len(d):
            md.append(f"| {c} | {np.mean(d < 0):.2f} | {np.mean(d == 0):.2f} | {np.mean((d > 0) & np.isfinite(d)):.2f} | "
                      f"{np.mean(~np.isfinite(d)):.2f} |")
    (out / "trazas.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out / 'trazas.md'}")


if __name__ == "__main__":
    main()
