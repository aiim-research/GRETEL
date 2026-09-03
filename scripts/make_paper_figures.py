#!/usr/bin/env python3
"""Regenerate the per-generator result figures for paper_reviewed.

One combined figure per generator (DCE, OFS, RSGG), 2x2 panels
(GED, FED, OC, Correctness), grouped bars over datasets, with:
  * five methods per group: generator-only, LBS, OBS, DDBS, RHC  [R1.3]
  * a single shared legend for the whole figure                  [R2.8]
  * FED shown only for the attributed datasets BBBP / Synthie    [R2.9]
  * the exact value printed on top of every bar
  * GED capped at GED_CAP on the axis: taller bars run off the top of the
    panel but their exact value is still annotated on the bar

Data come from lab/output/results/, read through the same aggregation the
notebook plots (scripts/_results_agg.py). Every bar, the generator-only one
included, is a real cell: the generator-only bar is the <ds>_<gen>_dummy
scope. Cells with no results yet are drawn as a hatched 'TODO' placeholder.

Outputs:  document/paper_reviewed/images/<gen>_results.{png,pdf}
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _results_agg import displayed_mean

PAPER_IMAGES = Path("/home/rodrigo/projects/GRETEL stuff/Documents/paper-reviewed/images")
LAB_GRAPHICS = Path(__file__).resolve().parent.parent / "lab" / "graphics"

GENS = ["dce", "ofs", "dfs", "rsgg"]
GEN_TITLE = {"dce": "DCE", "ofs": "OFS", "dfs": "DFS", "rsgg": "RSGG"}
# (display dataset label, scope dataset)
DATASETS = [("ASD", "asd"), ("TCR", "tcr-tco-300"), ("BBBP", "bbbp"), ("Synthie", "synthie")]
ATTRIBUTED = {"bbbp", "synthie"}

# (legend label, scope minimizer token)
METHODS = [("Generator only", "dummy"),
           ("LBS", "lcls"),
           ("OBS", "obs"),
           ("DBS", "dbs"),
           ("RHC", "rhc")]
PALETTE = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# (metric short, panel title, y label, attributed-only?, cap)
# Correctness is intentionally NOT a panel: it is a property of the generator,
# so it is shown once per dataset in the legend quadrant (see make_figure),
# not as a per-minimizer metric.
PANELS = [("GraphEditDistance", "GED (lower is better)", "GED", False, 100.0),
          ("FeatureEditDistance", "FED (lower is better)", "FED", True, None),
          ("OracleCalls", "Oracle Calls (lower is better)", "OC", False, None)]

GED_CAP = 100.0


def fmt(v, metric):
    if metric == "Correctness":
        return f"{v:.2f}"
    if metric == "OracleCalls":
        return f"{v:.0f}"
    return f"{v:.2f}" if v < 100 else f"{v:.1f}"


def cell_value(ds, gen, mk, metric):
    """Value for one bar, read from the result store.

    The 'Generator only' bar is the ``<ds>_<gen>_dummy`` scope (the pass-through
    minimizer), so it goes through the same aggregation as every other bar.
    Returns None when the cell has no results yet, which draws the hatched
    placeholder in :func:`draw_panel`."""
    v, n = displayed_mean(ds, gen, mk, metric)
    if v != v or n == 0:   # NaN or no folds
        return None
    return v


def draw_panel(ax, gen, metric, title, ylabel, attr_only, cap):
    dss = [(lab, ds) for lab, ds in DATASETS if (ds in ATTRIBUTED or not attr_only)]
    x = np.arange(len(dss))
    width = 0.16
    n_methods = len(METHODS)

    top = 0.0
    for mi, (mlabel, mk) in enumerate(METHODS):
        offs = x + (mi - (n_methods - 1) / 2) * width
        for xi, (lab, ds) in enumerate(dss):
            v = cell_value(ds, gen, mk, metric)
            ox = offs[xi]
            if v is None:
                # missing (e.g. RSGG generator-only not produced yet)
                ax.bar(ox, (cap or 1) * 0.04, width, color="#dddddd",
                       edgecolor="grey", hatch="///", linewidth=0.4)
                ax.text(ox, 0, "TODO", ha="center", va="bottom", rotation=90,
                        fontsize=6, color="grey")
                continue
            disp_h = v
            clipped = cap is not None and v > cap
            if clipped:
                disp_h = cap
            ax.bar(ox, disp_h, width, color=PALETTE[mi], edgecolor="black",
                   linewidth=0.4)
            top = max(top, disp_h)
            label_y = min(disp_h, (cap * 0.92) if cap else disp_h)
            txt = ("↑" + fmt(v, metric)) if clipped else fmt(v, metric)
            ax.text(ox, label_y, txt, ha="center", va="bottom", rotation=90,
                    fontsize=6.0)

    ax.set_xticks(x)
    ax.set_xticklabels([lab for lab, _ in dss], fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    if metric == "Correctness":
        ax.set_ylim(0, 1.18)
    elif cap is not None:
        ax.set_ylim(0, cap * 1.12)
    else:
        ax.set_ylim(0, top * 1.30 if top > 0 else 1)
    if attr_only:
        ax.text(0.5, -0.22, "FED omitted for attribute-free datasets (ASD, TCR).",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8, style="italic", color="#444444")


def make_figure(gen, out_dir=PAPER_IMAGES):
    out_dir = Path(out_dir)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    flat = axes.ravel()
    for ax, (metric, title, ylabel, attr_only, cap) in zip(flat, PANELS):
        draw_panel(ax, gen, metric, title, ylabel, attr_only, cap)

    # the spare quadrant holds the shared legend and the generator correctness
    legend_ax = flat[len(PANELS)]
    legend_ax.axis("off")
    handles = [Patch(facecolor=PALETTE[i], edgecolor="black", label=lab)
               for i, (lab, _) in enumerate(METHODS)]
    legend_ax.legend(handles=handles, loc="upper center", ncol=1, frameon=True,
                     fontsize=12, title="Method", title_fontsize=12)

    # Correctness is a property of the generator (identical across minimizers),
    # so instead of a separate table it is shown here, one value per dataset
    # (from the LBS runs).
    bits = []
    for lab, ds in DATASETS:
        c, nf = displayed_mean(ds, gen, "lcls", "Correctness")
        bits.append(f"{lab}: {c:.2f}" if (nf and c == c) else f"{lab}: -")
    corr_txt = ("Correctness (generator)\n"
                + "      ".join(bits[:2]) + "\n" + "      ".join(bits[2:]))
    legend_ax.text(0.5, 0.05, corr_txt, transform=legend_ax.transAxes,
                   ha="center", va="bottom", fontsize=10,
                   bbox=dict(boxstyle="round", facecolor="#f5f5f5",
                             edgecolor="#cccccc"))
    fig.suptitle(f"{GEN_TITLE[gen]} results", fontsize=14, y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{gen}_results.{ext}", dpi=150)
    plt.close(fig)
    print(f"wrote {out_dir / (gen + '_results.png')}")


def generate_all(out_dir=PAPER_IMAGES, gens=None):
    """Generate the per-generator figures from the result store."""
    for gen in (gens or GENS):
        make_figure(gen, out_dir=out_dir)


def main():
    only = [a for a in sys.argv[1:] if a in GENS] or GENS
    for gen in only:
        make_figure(gen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
