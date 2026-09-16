#!/usr/bin/env python3
"""Generate the per-metric grouped-bar figures for the thesis' "Resultados
finales" section.

Reads aggregated metrics from ``lab/output/results/<scope>/...`` (where
``scope == <dataset>_<gen>_<variant>``), selects the three pipelines that
participate in the final comparison —

    * dcem-bls-snt  →  scope ``<dataset>_dcm_lcls-net-trainable``
    * OFS+OBS (= OBS-explainer)  →  scope ``<dataset>_ofs_obs``
    * DFS+DBS (= DDBS)  →  scope ``<dataset>_dfs_dbs``

— and produces **four** standalone grouped bar charts (GED, FED, OC,
Correctness) covering all eight datasets in the same order as the tables
in Results.tex.

Missing combos (still in flight) render as a hatched grey bar with a small
"—" annotation so the figure stays usable while you wait for the rest of
the folds. Re-run the script after each batch finishes to refresh.

Outputs (one PDF + PNG per metric):
    /home/rodrigo/projects/GRETEL stuff/document/images/final_results_ged.{pdf,png}
    /home/rodrigo/projects/GRETEL stuff/document/images/final_results_fed.{pdf,png}
    /home/rodrigo/projects/GRETEL stuff/document/images/final_results_oc.{pdf,png}
    /home/rodrigo/projects/GRETEL stuff/document/images/final_results_correctness.{pdf,png}

Usage::

    python scripts/plot_final_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO / "lab" / "output" / "results"
THESIS_IMAGES = Path("/home/rodrigo/projects/GRETEL stuff/document/images")

DATASETS = ["synthie", "bbbp", "enzymes", "bzr", "aids", "proteins", "colors-3", "asd"]
DATASET_LABELS = {
    "synthie":  "Synthie",
    "bbbp":     "BBBP",
    "enzymes":  "ENZYMES",
    "bzr":      "BZR",
    "aids":     "AIDS",
    "proteins": "PROTEINS",
    "colors-3": "COLORS-3",
    "asd":      "ASD",
}

METHODS = [
    ("dcem-bls-snt",   "dcm_lcls-net-trainable"),
    ("OBS (ofs-obs)",  "ofs_obs"),
    ("DDBS (dfs-dbs)", "dfs_dbs"),
]

# (json-key-substring, output-stem, axis label, title, log-scale?, ymax_override)
METRICS = [
    ("GraphEditDistance",   "final_results_ged",          "GED",          "Graph Edit Distance (menor es mejor)",   False, None),
    ("FeatureEditDistance", "final_results_fed",          "FED",          "Feature Edit Distance (menor es mejor)", False, None),
    ("OracleCalls",         "final_results_oc",           "OC (log)",     "Oracle Calls (menor es mejor)",          True,  None),
    ("Correctness",         "final_results_correctness",  "Correctness",  "Correctness (mayor es mejor)",           False, 1.05),
]


def _read_aggregated(scope: str) -> dict[str, float]:
    """Return mean of all metrics across folds for a scope, empty dict if no data."""
    scope_dir = RESULTS_ROOT / scope
    if not scope_dir.is_dir():
        return {}
    values: dict[str, list[float]] = {m[0]: [] for m in METRICS}
    for f in scope_dir.rglob("results_*_*.json"):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        for stage_key, entries in data.get("results", {}).items():
            for metric_key in values:
                if metric_key in stage_key:
                    for entry in entries:
                        v = entry.get("value")
                        if isinstance(v, (int, float)):
                            values[metric_key].append(float(v))
                    break
    return {k: float(np.mean(arr)) for k, arr in values.items() if arr}


def main() -> int:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; pip install matplotlib first")
        return 2

    # Collect once.
    data: dict[str, dict[str, dict[str, float]]] = {}
    for ds in DATASETS:
        data[ds] = {}
        for method_label, variant in METHODS:
            data[ds][method_label] = _read_aggregated(f"{ds}_{variant}")

    THESIS_IMAGES.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(DATASETS))
    width = 0.27
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for metric, stem, ylabel, title, logscale, ymax_override in METRICS:
        fig, ax = plt.subplots(figsize=(14, 6))
        for idx, (method_label, _) in enumerate(METHODS):
            heights, missing = [], []
            for ds in DATASETS:
                v = data[ds][method_label].get(metric)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    heights.append(np.nan)
                    missing.append(True)
                else:
                    heights.append(v)
                    missing.append(False)
            offsets = x + (idx - 1) * width
            real = np.array([h if not m else 0.0 for h, m in zip(heights, missing)])
            bars = ax.bar(offsets, real, width, color=palette[idx],
                          edgecolor="black", linewidth=0.5, label=method_label)
            for off, h, m in zip(offsets, heights, missing):
                if m:
                    # hatched placeholder
                    ax.bar([off], [1e-6 if logscale else 0.0], width,
                           color="#dddddd", edgecolor="grey",
                           hatch="///", linewidth=0.4)
                    base = 1 if logscale else 0.0
                    ax.text(off, base, "—", ha="center", va="bottom",
                            fontsize=9, color="grey")
                else:
                    # numeric annotation on top of each bar
                    fmt = f"{h:.2f}" if h < 100 else (
                          f"{h:.1f}" if h < 10000 else f"{h:.0f}")
                    ax.text(off, h, fmt, ha="center",
                            va="bottom", fontsize=8, rotation=0)

        ax.set_title(title, fontsize=13)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS],
                           rotation=20, ha="right", fontsize=11)
        ax.grid(True, axis="y", linestyle="--", alpha=0.45)

        if logscale:
            ax.set_yscale("log")
            real_values = [v for ds in DATASETS for m, _ in METHODS
                           for v in [data[ds][m].get(metric)] if v]
            ax.set_ylim(1, (max(real_values) * 1.6) if real_values else 1e5)
        else:
            if ymax_override is not None:
                ax.set_ylim(0, ymax_override)
            else:
                real_values = [v for ds in DATASETS for m, _ in METHODS
                               for v in [data[ds][m].get(metric)] if v]
                if real_values:
                    ax.set_ylim(0, max(real_values) * 1.18)

        ax.legend(loc="upper right", frameon=True, fontsize=10)
        fig.tight_layout()
        fig.savefig(THESIS_IMAGES / f"{stem}.pdf")
        fig.savefig(THESIS_IMAGES / f"{stem}.png", dpi=150)
        plt.close(fig)
        print(f"wrote {THESIS_IMAGES / (stem + '.pdf')}")

    # Coverage report.
    print("\nData coverage per (dataset, method):")
    for ds in DATASETS:
        for label, _ in METHODS:
            metrics = data[ds][label]
            tag = ", ".join(f"{m[0]}={metrics[m[0]]:.3f}" if m[0] in metrics else f"{m[0]}=—"
                            for m in METRICS)
            print(f"  {ds:<10} | {label:<18} | {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
