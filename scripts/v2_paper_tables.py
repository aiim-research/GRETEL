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
ARMS = [("DCEM", "dummy"), ("OBS", "obs"), ("RHC", "rhc"), ("bls_v0", "lcls-v0"), ("bls_v1-F", "lcls-v1-f"),
        ("control-F", "lcls-sel-v1-untrained-f"), ("selector_v1-F", "lcls-sel-v1-f"),
        ("RHC-B", "rhc-b")]
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
         ("RHC-B", "selector_v1-F", ["struct", "feat", "total", "calls"])]


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
    return {"n": len(ids), "rel": x.mean() / max(y.mean(), 1e-9) - 1, "lo": np.quantile(r, .025), "hi": np.quantile(r, .975),
            "p": p, "better": int((d < 0).sum()), "ties": int((d == 0).sum()), "worse": int((d > 0).sum())}


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
    args = ap.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    results = Path(args.results)
    rng = np.random.default_rng(0)
    data = {c: {n: load(results, f"v2-{c}-sel_dcm_{s}") for n, s in ARMS} for c in CELLS}

    cols = ["cell", "dataset", "oracle", "arm", "fold", "instance_id", "ok", "struct", "feat", "total", "ged", "calls",
            "runtime", "sparsity"]
    with open(out / "por_instancia.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for c in CELLS:
            for name, _ in ARMS:
                for (fold, iid), v in sorted(data[c][name].items(), key=lambda t: (t[0][0], int(t[0][1]))):
                    w.writerow([c, c.removesuffix("-m70"), "margin 0.7" if c.endswith("-m70") else "base", name, fold, iid]
                               + [round(v.get(k, float("nan")), 6) for k in cols[6:]])

    md = ["# Tablas del protocolo v2, 12 celdas clásicas, DCEM, folds 0 a 9", "",
          "Generado por `scripts/v2_paper_tables.py` (GRETEL). Costes medios sobre las instancias correctas, media y "
          "desviación típica sobre las medias de los 10 folds. S estructura (StructuralEditDistance), F features "
          "(MatchedFeatureDistance), total S + F (EditDistance), GED la fórmula exacta del paper 1, llamadas al oráculo.", ""]
    md += ["## 1. Media por celda y brazo", "",
           "| Celda | Brazo | n | correctas | S | F | total | GED | llamadas |", "|---|---|---|---|---|---|---|---|---|"]
    for c in CELLS:
        for name, _ in ARMS:
            d = data[c][name]
            ok = np.mean([v.get("ok", 0) >= 1 for v in d.values()])
            cells = [f"{m:.2f} ± {s:.2f}" for m, s in (fold_stats(d, k) for k in ("struct", "feat", "total", "ged"))]
            cm, cs = fold_stats(d, "calls")
            md.append(f"| {c} | {name} | {len(d)} | {ok:.3f} | " + " | ".join(cells) + f" | {cm:.0f} ± {cs:.0f} |")
    md += ["", "### Cuantiles del total (instancias correctas)", "",
           "Mediana, percentil 75 y percentil 90 del total S + F sobre todas las instancias de la celda. Separa los "
           "brazos de cola pesada (RHC, RHC-B) de los que fallan poco.", "",
           "| Celda | Brazo | mediana | p75 | p90 | máximo |", "|---|---|---|---|---|---|"]
    for c in CELLS:
        for name, _ in ARMS:
            x = np.array([v["total"] for v in data[c][name].values() if v.get("ok", 0) >= 1])
            if len(x):
                md.append(f"| {c} | {name} | " + " | ".join(f"{q:.2f}" for q in np.percentile(x, [50, 75, 90, 100])) + " |")
    md += ["", "## 2. Comparaciones pareadas", "",
           "Diferencia relativa de las medias (A/B - 1), intervalo bootstrap al 95 % entre corchetes, y `*` si el Wilcoxon "
           "pareado sigue por debajo de 0,05 tras Holm sobre las 12 celdas. Instancias correctas en los dos brazos.", ""]
    for a, b, keys in PAIRS:
        res = {k: {c: pair(data[c][a], data[c][b], k, rng) for c in CELLS} for k in keys}
        for k in keys:
            adj = holm([res[k][c]["p"] for c in CELLS])
            for c, p in zip(CELLS, adj):
                res[k][c]["ph"] = p
        md += [f"### {a} frente a {b}", "", "| Celda | " + " | ".join(keys) + " | gana/empata/pierde (total o S) |",
               "|---|" + "---|" * (len(keys) + 1)]
        for c in CELLS:
            row = [f"{res[k][c]['rel']:+.1%} [{res[k][c]['lo']:+.0%}, {res[k][c]['hi']:+.0%}]{'*' if res[k][c]['ph'] < .05 else ''}"
                   for k in keys]
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
    (out / "tablas.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out / 'por_instancia.csv'} and {out / 'tablas.md'}")


if __name__ == "__main__":
    main()
