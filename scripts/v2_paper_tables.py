#!/usr/bin/env python3
"""The selector paper's numbers on protocol v2 (docs/v2/RUNS.md): the 12 classic cells (six datasets, each with its
margin oracle ``-m70``), DCEM seeds, folds 0 to 9. The triangle cells are left out (a study of their own).

Writes, into OUT:

* ``por_instancia.csv``: one row per cell, arm, fold and instance, with every metric (correctness, structural and
  feature cost and their sum, paper 1's GED, oracle calls, runtime, sparsity);
* ``tablas.md``: per cell and arm the mean and standard deviation over the ten fold means (correct instances only),
  the paired comparisons up the ladder per cost component (relative difference of the means, 95 percent bootstrap
  interval, Wilcoxon p with Holm across the 12 cells), paper 1's main table against the published one, and the
  selector's pretraining reports.

    python scripts/v2_paper_tables.py OUT [--results lab/output/results]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parent.parent
DATASETS = ["aids", "bbbp", "bzr", "enzymes", "proteins", "synthie"]
CELLS = DATASETS + [d + "-m70" for d in DATASETS]
# COLORS-3 (--colors): its two cells, reported apart in the appendix (Holm across its two cells).
COLORS = ["colors-3-s20", "colors-3-m70-s20"]
PAPER1_COLORS = {"colors-3-s20": (92.96, 9.18, 82.96)}
# (name, scope after "v2-<cell>-sel_"). Arm E (2026-11-15): DBS on the DCEM seeds. The paper compares minimizers on
# the DCEM seeds only (decision of 2026-11-17): the end-to-end explainers OFS with obs (``ofs_obs``) and DDBS
# (``dfs_dbs``) ran in arm E but stay out of the tables (they never edit features, and solve 7 to 100 % of instances).
ARMS = [("DCEM", "dcm_dummy"), ("OBS", "dcm_obs"), ("RHC", "dcm_rhc"), ("bls_v0", "dcm_lcls-v0"), ("bls_v1-F", "dcm_lcls-v1-f"),
        ("control-F", "dcm_lcls-sel-v1-untrained-f"), ("selector_v1-F", "dcm_lcls-sel-v1-f"),
        ("RHC-B", "dcm_rhc-b"), ("DBS", "dcm_dbs")]
# The three (cell, arm, fold) whose original run did not reproduce (arm D): the tables take the traced rerun ``-t``
# instead, which is the reproducible one (arm E: the two deterministic repetitions of the AIDS fold equal it, 200/200).
RERUN = {("aids", "control-F", 9), ("synthie", "selector_v1-F", 9), ("synthie-m70", "selector_v1-F", 2)}
# The stability table (arm E): selector_v1-F with seeds 1 and 2 next to seed 0, each against bls_v1-F (seed 0).
SEEDS = [("selector_v1-F", "dcm_lcls-sel-v1-f")]
REF = ("bls_v1-F", "dcm_lcls-v1-f")
KEYS = {"ok": "Correctness", "struct": "StructuralEditDistance", "feat": "MatchedFeatureDistance",
        "ged": "GraphEditDistance", "calls": "OracleCalls", "runtime": "Runtime", "sparsity": "Sparsity"}
PAPER1 = {"synthie": (201.41, 2.16, 185.22), "bbbp": (18.98, 3.20, 14.43), "enzymes": (58.46, 2.55, 14.46),
          "bzr": (38.06, 2.06, 28.79), "aids": (17.25, 13.02, 16.92), "proteins": (73.19, 50.18, 64.58)}
PAIRS = [("bls_v0", "OBS", ["struct", "ged", "calls"]), ("bls_v0", "RHC", ["struct", "ged", "calls"]),
         ("bls_v1-F", "bls_v0", ["struct", "feat", "total", "calls"]),
         ("bls_v1-F", "OBS", ["struct", "feat", "total", "calls"]), ("bls_v1-F", "RHC", ["struct", "feat", "total", "calls"]),
         ("selector_v1-F", "bls_v1-F", ["struct", "feat", "total", "calls"]),
         ("control-F", "bls_v1-F", ["struct", "feat", "total", "calls"]),
         ("selector_v1-F", "control-F", ["struct", "feat", "total", "calls"]),
         ("selector_v1-F", "RHC", ["struct", "feat", "total", "calls"]),
         ("RHC-B", "bls_v1-F", ["struct", "feat", "total", "calls"]),
         ("RHC-B", "selector_v1-F", ["struct", "feat", "total", "calls"]),
         ("bls_v0", "DBS", ["struct", "ged", "calls"]), ("DBS", "OBS", ["struct", "ged", "calls"])]


def load(results: Path, scope: str) -> dict:
    out = {}
    for f in glob.glob(str(results / scope / "**" / "results_*_*.json"), recursive=True):
        fold = int(re.search(r"results_(\d+)_", f).group(1))
        res = json.loads(Path(f).read_text())["results"]
        for short, key in KEYS.items():
            full = next((k for k in res if k.endswith("." + key)), None)
            for e in res.get(full, []):
                out.setdefault((fold, str(e["id"])), {})[short] = float(e["value"])
    for v in out.values():
        v["total"] = v.get("struct", 0.0) + v.get("feat", 0.0)
    return out


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


def pair(a, b, key, rng):
    ids = [i for i in a if i in b and a[i].get("ok", 0) >= 1 and b[i].get("ok", 0) >= 1]
    x = np.array([a[i][key] for i in ids]); y = np.array([b[i][key] for i in ids]); d = x - y
    p = float(wilcoxon(x, y).pvalue) if np.any(d != 0) else 1.0
    bs = rng.integers(0, len(ids), (2000, len(ids)))
    r = x[bs].mean(1) / np.maximum(y[bs].mean(1), 1e-9) - 1
    return {"n": len(ids), "a": x.mean(), "b": y.mean(), "rel": x.mean() / max(y.mean(), 1e-9) - 1, "lo": np.quantile(r, .025), "hi": np.quantile(r, .975),
            "p": p, "better": int((d < 0).sum()), "ties": int((d == 0).sum()), "worse": int((d > 0).sum())}


def cell(r: dict) -> str:
    """A/B - 1 with its interval; the means themselves when B is zero (a minimizer that never changes a feature)."""
    star = "*" if r["ph"] < .05 else ""
    if r["b"] < 1e-6:
        return f"{r['a']:.2f} frente a 0{star}"
    return f"{r['rel']:+.1%} [{r['lo']:+.0%}, {r['hi']:+.0%}]{star}"


def fold_stats(d: dict, key: str):
    means = [np.mean([v[key] for (f, _), v in d.items() if f == fold and v.get("ok", 0) >= 1])
             for fold in sorted({f for f, _ in d})]
    means = [m for m in means if not np.isnan(m)]
    return float(np.mean(means)), float(np.std(means))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--results", default=str(REPO / "lab/output/results"))
    ap.add_argument("--pretrained", default=str(REPO / "models/pretrained/v2"))
    ap.add_argument("--colors", action="store_true", help="the two COLORS-3 cells instead of the 12 classic ones, "
                    "written as tablas_colors.md and por_instancia_colors.csv")
    args = ap.parse_args(argv)
    global CELLS, PAPER1
    suffix = ""
    if args.colors:
        CELLS, PAPER1, suffix = COLORS, PAPER1_COLORS, "_colors"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    results = Path(args.results)
    rng = np.random.default_rng(0)
    data = {c: {n: load(results, f"v2-{c}-sel_{s}") for n, s in ARMS} for c in CELLS}
    for c, n, fold in (r for r in RERUN if r[0] in CELLS):
        rerun = {k: v for k, v in load(results, f"v2-{c}-sel_{dict(ARMS)[n]}-t").items() if k[0] == fold}
        if rerun:
            data[c][n] = {k: v for k, v in data[c][n].items() if k[0] != fold} | rerun
    present = [n for n, _ in ARMS if any(data[c][n] for c in CELLS)]      # arms with results (arm E arrives later)

    cols = ["cell", "dataset", "oracle", "arm", "fold", "instance_id", "ok", "struct", "feat", "total", "ged", "calls",
            "runtime", "sparsity"]
    with open(out / f"por_instancia{suffix}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for c in CELLS:
            for name in present:
                for (fold, iid), v in sorted(data[c][name].items(), key=lambda t: (t[0][0], int(t[0][1]))):
                    w.writerow([c, c.removesuffix("-m70"), "margin 0.7" if c.endswith("-m70") else "base", name, fold, iid]
                               + [round(v.get(k, float("nan")), 6) for k in cols[6:]])

    md = [f"# Tablas del protocolo v2, {'COLORS-3 (dos celdas)' if args.colors else '12 celdas clásicas'}, DCEM, folds 0 a 9", "",
          "Generado por `scripts/v2_paper_tables.py` (GRETEL). Costes medios sobre las instancias correctas, media y "
          "desviación típica sobre las medias de los 10 folds. S estructura (StructuralEditDistance), F features "
          "(MatchedFeatureDistance), total S + F (EditDistance), GED la fórmula exacta del paper 1, llamadas al oráculo. "
          "Tres folds cuya ejecución original no se reprodujo usan su repetición trazada `-t`, que es la reproducible: "
          + ", ".join(f"{n} en {c} fold {f}" for c, n, f in sorted(RERUN)) + ".", ""]
    md += ["## 1. Media por celda y brazo", "",
           "| Celda | Brazo | n | correctas | S | F | total | GED | llamadas |", "|---|---|---|---|---|---|---|---|---|"]
    for c in CELLS:
        for name in present:
            d = data[c][name]
            if not d:
                continue
            ok = np.mean([v.get("ok", 0) >= 1 for v in d.values()])
            cells = [f"{m:.2f} ± {s:.2f}" for m, s in (fold_stats(d, k) for k in ("struct", "feat", "total", "ged"))]
            cm, cs = fold_stats(d, "calls")
            md.append(f"| {c} | {name} | {len(d)} | {ok:.3f} | " + " | ".join(cells) + f" | {cm:.0f} ± {cs:.0f} |")
    md += ["", "### Cuantiles del total (instancias correctas)", "",
           "Mediana, percentil 75 y percentil 90 del total S + F sobre todas las instancias de la celda. Separa los "
           "brazos de cola pesada (RHC, RHC-B) de los que fallan poco.", "",
           "| Celda | Brazo | mediana | p75 | p90 | máximo |", "|---|---|---|---|---|---|"]
    for c in CELLS:
        for name in present:
            x = np.array([v["total"] for v in data[c][name].values() if v.get("ok", 0) >= 1])
            if len(x):
                md.append(f"| {c} | {name} | " + " | ".join(f"{q:.2f}" for q in np.percentile(x, [50, 75, 90, 100])) + " |")
    md += ["", "## 2. Comparaciones pareadas", "",
           "Diferencia relativa de las medias (A/B - 1), intervalo bootstrap al 95 % entre corchetes, y `*` si el Wilcoxon "
           "pareado sigue por debajo de 0,05 tras Holm sobre las celdas de la tabla. Instancias correctas en los dos brazos.", ""]
    for a, b, keys in PAIRS:
        if not all(data[c][a] and data[c][b] for c in CELLS):
            continue                     # an arm without results in some cell (arm E not imported yet)
        res = {k: {c: pair(data[c][a], data[c][b], k, rng) for c in CELLS} for k in keys}
        for k in keys:
            adj = holm([res[k][c]["p"] for c in CELLS])
            for c, p in zip(CELLS, adj):
                res[k][c]["ph"] = p
        md += [f"### {a} frente a {b}", "", "| Celda | " + " | ".join(keys) + " | gana/empata/pierde (total o S) |",
               "|---|" + "---|" * (len(keys) + 1)]
        for c in CELLS:
            row = [cell(res[k][c]) for k in keys]
            ref = res["total" if "total" in keys else "struct"][c]
            md.append(f"| {c} | " + " | ".join(row) + f" | {ref['better']}/{ref['ties']}/{ref['worse']} |")
        md.append("")
    md += ["## 3. Tabla principal del paper 1 (DCEM solo, con BLS = bls_v0, con OBS)", "",
           "Publicado: media de 10 folds del paper 1 (protocolo v1, con padding, GED estructural). v2: S y GED del "
           "protocolo nuevo, media ± desviación sobre los folds.", "",
           "| Dataset | DCEM pub. | DCEM v2 S | DCEM v2 GED | BLS pub. | bls_v0 v2 S | bls_v0 v2 GED | OBS pub. | OBS v2 S | OBS v2 GED |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for ds, (g, l, o) in PAPER1.items():
        row = [ds]
        for pub, arm in ((g, "DCEM"), (l, "bls_v0"), (o, "OBS")):
            s = fold_stats(data[ds][arm], "struct"); e = fold_stats(data[ds][arm], "ged")
            row += [f"{pub:.2f}", f"{s[0]:.2f} ± {s[1]:.2f}", f"{e[0]:.2f} ± {e[1]:.2f}"]
        md.append("| " + " | ".join(row) + " |")
    md += ["", "## 4. Preentrenamiento del selector (checkpoints sel2fm-pre)", "",
           "`-f1` se entrena con la colección del fold 1 y sirve a los folds 0 y 2 a 9; `-f2`, con la del fold 2, sirve al "
           "fold 1. AUC de predicción de vuelco en validación (instancias apartadas) y skill = 1 - logloss / logloss de la "
           "tasa base.", "",
           "| Checkpoint | ejemplos | quitar: tasa base | quitar: AUC | quitar: skill | añadir: tasa base | añadir: AUC | añadir: skill |",
           "|---|---|---|---|---|---|---|---|"]
    for c in CELLS:
        for f in ("f1", "f2"):
            p = Path(args.pretrained) / f"edge_selector_sel2fm-pre_{c}-{f}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            row = [f"{c}-{f}", str(d["examples"])]
            for m in ("remove", "add"):
                v = d["modes"].get(m, {}); fin = v.get("final", {})
                row += [f"{v.get('base_rate', float('nan')):.3f}", f"{fin.get('auc_flip', float('nan')):.3f}",
                        f"{fin.get('skill', float('nan')):+.3f}"]
            md.append("| " + " | ".join(row) + " |")
    seeds = {(n, k): {c: data[c][n] if k == 0 else load(results, f"v2-{c}-sel_{sc}-s{k}") for c in CELLS}
             for n, sc in SEEDS for k in (0, 1, 2)}     # seed 0 is the fleet's run, with the RERUN folds
    if any(seeds[(n, 1)][c] for n, _ in SEEDS for c in CELLS):
        md += ["", "## 5. Estabilidad frente a la semilla del minimizador (semillas 0, 1 y 2)", "",
               "Total S + F medio por semilla (instancias correctas, los diez folds) y su media ± desviación entre "
               "semillas. Diferencia relativa de cada semilla del selector frente a bls_v1-F (semilla 0): la "
               "significación es Wilcoxon pareado con Holm sobre las celdas, como en la sección 2.", "",
               "| Celda | Brazo | semilla 0 | semilla 1 | semilla 2 | media ± desv. |", "|---|---|---|---|---|---|"]
        for c in CELLS:
            for n, _ in SEEDS:
                m = [np.mean([v["total"] for v in seeds[(n, k)][c].values() if v.get("ok", 0) >= 1]) if seeds[(n, k)][c]
                     else float("nan") for k in (0, 1, 2)]
                md.append(f"| {c} | {n} | " + " | ".join(f"{x:.2f}" for x in m) + f" | {np.nanmean(m):.2f} ± {np.nanstd(m):.2f} |")
        md += ["", "| Celda | selector frente a bls_v1-F, semilla 0 | semilla 1 | semilla 2 |", "|---|---|---|---|"]
        ref = {c: data[c][REF[0]] for c in CELLS}
        res = {k: {c: pair(seeds[("selector_v1-F", k)][c], ref[c], "total", rng) if seeds[("selector_v1-F", k)][c] else None
                   for c in CELLS} for k in (0, 1, 2)}
        for k in (0, 1, 2):
            ok = [c for c in CELLS if res[k][c]]
            for c, p in zip(ok, holm([res[k][c]["p"] for c in ok])):
                res[k][c]["ph"] = p
        for c in CELLS:
            md.append(f"| {c} | " + " | ".join(cell(res[k][c]) if res[k][c] else "" for k in (0, 1, 2)) + " |")
    (out / f"tablas{suffix}.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out / f'por_instancia{suffix}.csv'} and {out / f'tablas{suffix}.md'}")


if __name__ == "__main__":
    main()
