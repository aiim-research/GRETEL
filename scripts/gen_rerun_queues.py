#!/usr/bin/env python3
"""Queues to rerun every thesis and paper-revision experiment in the current result format.

Results before the BLS selector were kept as metrics only and were moved out of the repository
(docs/reproducing-experiments.md, section 4), so the experiments of the LBS thesis
(docs/thesis-experiments.md) and of the paper revision (docs/revision/REVISION_EXPERIMENTS.md) run
again, now with the per-instance counterfactual dumps and the committed caches. The old selector
(thesis section 5.5, `*-lcls-net-trainable`) is left out: the new selector replaces it.

Two things are written:

* every config of the cells below gets ``generator_cache`` (the decoupled protocol the paper states:
  each generator runs once per fold and every minimizer starts from the same counterfactual, with the
  generator's oracle calls added back). Idempotent; it changes the explainer's hash, so these results
  get new directories.
* the queues for ``launchers/slurm_submit.sh``. Phase A holds the generator-only runs (``*-dummy``),
  which fill the generator cache; the later phases reuse it, so a cell's phase A must be over before its
  other runs start (two runs that computed the same generator output at the same time could start from
  different counterfactuals, since OFS and DFS are not deterministic across runs).

  - ``docs/rerun/light-1-generators.md`` and ``light-2-minimizers.md``: every cell of the datasets with
    an exact or rule-based oracle and few, small instances (``LIGHT``: 10 s to 2 min per fold measured on
    a laptop), phase A and the rest, submitted together with ``slurm_submit.sh -c`` (the second array
    starts when the first has ended).
  - ``docs/rerun/Q<nn>.md``: everything else, queues of at most 30 configs, one at a time. A queue never
    mixes phases, and the cells that need caches built on the server first (``NEEDS_PREP``: the BZR and
    ENZYMES datasets and oracles, and the DCM of PROTEINS, BZR and ENZYMES) come last in each phase.

    python scripts/gen_rerun_queues.py
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TREE = "lab/config/generate_minimize"
OUT = REPO / "docs" / "rerun"
CACHE = "lab/data/cache/generator_outputs"
QUEUE_SIZE = 30
LIGHT = {"tcr-500-28", "tcr-tco-300", "asd"}


def needs_prep(ds: str, gen: str) -> bool:
    return ds in ("bzr", "enzymes") or (ds == "proteins" and gen == "dcm")

PHASES = {
    "A": "generator alone (`*-dummy`), fills the generator cache",
    "B": "thesis main results: BLS and OBS-min (PI1 to PI3, DCEM tables)",
    "C": "thesis ablations and baselines: budget (PI4), attributes (PI5), neighbourhood (PI6), final-results baselines",
    "D": "paper revision only: DBS, RHC, DFS, Tree-Cycles 300 (E1), LBS seeds (E1b), strategy order (E3)",
}


def cells() -> dict[tuple[str, str, str], str]:
    """(dataset, generator dir, variant dir) -> phase, the earliest phase that needs the cell."""
    out: dict[tuple[str, str, str], str] = {}

    def add(phase, ds, gen, var):
        if var.endswith("-dummy"):
            phase = "A"
        key = (ds, gen, var)
        out[key] = min(out.get(key, phase), phase)

    # Thesis, PI1 to PI3: three generators, four datasets, BLS / OBS-min / generator alone.
    for ds, g, m in itertools.product(["asd", "tcr-500-28", "bbbp", "synthie"], ["dce", "ofs", "rsgg"],
                                      ["dummy", "lcls", "obs"]):
        add("B", ds, g, f"{g}-{m}")
    # Thesis, DCEM tables: DCE and DCEM on eight datasets.
    for ds, g, m in itertools.product(["synthie", "bbbp", "enzymes", "bzr", "aids", "colors-3", "proteins", "asd"],
                                      ["dce", "dcm"], ["dummy", "lcls", "obs"]):
        add("B", ds, g, f"{g}-{m}")
    # Thesis, PI4: Synthie under six oracle-call budgets.
    for b, m in itertools.product(["2.5", "5", "7.5", "10", "15", "20"], ["lcls", "obs"]):
        add("C", "synthie", f"dce-{b}", f"dce-{m}")
    # Thesis, PI5: the same minimizer without attribute manipulation.
    for ds, g in itertools.product(["synthie-no-att", "bbbp-no-attr"], ["ofs", "dce", "rsgg"]):
        add("C", ds, g, f"{g}-lcls")
    # Thesis, PI6: BLS restricted to deletions (var-1) or to swap and add (var-2).
    for ds, v in itertools.product(["bbbp", "synthie", "tcr-500-28", "asd"], ["var-1", "var-2"]):
        add("C", ds, "dce", f"dce-lcls-{v}")
    # Thesis, final results: the two coupled baselines (the integrated explainer used the old selector).
    for ds in ["synthie", "bbbp", "enzymes", "bzr", "aids", "asd"]:
        add("C", ds, "ofs", "ofs-obs")
        add("C", ds, "dfs", "dfs-dbs")
    # Revision, E1: four datasets, four generators, every minimizer.
    for ds, g, m in itertools.product(["asd", "bbbp", "synthie", "tcr-tco-300"], ["ofs", "dce", "rsgg", "dfs"],
                                      ["dummy", "obs", "dbs", "lcls", "rhc"]):
        add("D", ds, g, f"{g}-{m}")
    # Revision, E1b (LBS seeds) and E3 (strategy order).
    for ds in ["asd", "bbbp", "synthie", "tcr-tco-300"]:
        for s in (1, 2, 3):
            add("D", ds, "dce", f"dce-lcls-seed{s}")
        for v in ("var-3", "var-4"):
            add("D", ds, "dce", f"dce-lcls-{v}")
    return out


def _object_end(text: str, start: int) -> int:
    """Index just past the JSON object that opens at ``text[start] == "{"``."""
    depth, i, in_str = 0, start, False
    while True:
        c = text[i]
        if in_str:
            if c == "\\":
                i += 1
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1


def enable_cache(path: Path) -> bool:
    """Add ``generator_cache`` right after the minimizer, as text: half of these configs are formatted by
    hand, and rewriting them through json.dumps would reformat every line."""
    text = path.read_text()
    expl = json.loads(text)["doe-triplets"][0]["explainer"]["parameters"]
    if expl.get("generator_cache") == CACHE:
        return False
    key = text.index('"minimizer"', text.index("generate_minimize.GenerateMinimize"))
    end = _object_end(text, text.index("{", key))
    new = text[:end] + f', "generator_cache": "{CACHE}"' + text[end:]
    assert json.loads(new)["doe-triplets"][0]["explainer"]["parameters"]["generator_cache"] == CACHE, path
    path.write_text(new)
    return True


BLS_MODULES = ("local_search.local_search.", "local_search.local_search_var_1.", "local_search.local_search_var_2.",
               "local_search.local_search_var_3.", "local_search.local_search_var_4.")
SEVEN = ["identity", "average_smoothing", "weighted_smoothing", "laplacian_regularization", "feature_aggregation",
         "heat_kernel_diffusion", "random_walk_diffusion"]
V0_FIX = ('"stop_after_failed_iteration": true, "attribute_methods": [' + ", ".join(f'"{m}"' for m in SEVEN) + "], ")


def enable_v0_fix(path: Path) -> bool:
    """BLS v0 as the paper describes it (docs/selector/ANALISIS_CONTROLES.md, 5b): stop after the first outer
    iteration without improvement and the seven attribute strategies, stated in every BLS config (the removal
    level fix is in the code). As text, like ``enable_cache``; idempotent."""
    text = path.read_text()
    mini = json.loads(text)["doe-triplets"][0]["explainer"]["parameters"]["minimizer"]
    if not any(mod in mini["class"] for mod in BLS_MODULES):
        return False
    if mini["parameters"].get("stop_after_failed_iteration") and mini["parameters"].get("attribute_methods") == SEVEN:
        return False
    start = text.index('"minimizer"', text.index("generate_minimize.GenerateMinimize"))
    end = _object_end(text, text.index("{", start))
    key = text.index('"parameters"', start, end)
    brace = text.index("{", key, end)
    new = text[:brace + 1] + V0_FIX + text[brace + 1:]
    par = json.loads(new)["doe-triplets"][0]["explainer"]["parameters"]["minimizer"]["parameters"]
    assert par["stop_after_failed_iteration"] is True and par["attribute_methods"] == SEVEN, path
    path.write_text(new)
    return True


def main() -> int:
    by_phase: dict[str, list[str]] = {p: [] for p in PHASES}
    changed = fixed = 0
    for (ds, gen, var), phase in sorted(cells().items(), key=lambda kv: (kv[1], kv[0])):
        for fold in range(10):
            rel = f"{TREE}/{ds}/{gen}/{var}/generate_minimize{fold}.jsonc"
            changed += enable_cache(REPO / rel)
            fixed += enable_v0_fix(REPO / rel)
            by_phase[phase].append(rel)

    OUT.mkdir(parents=True, exist_ok=True)
    for old in list(OUT.glob("Q*.md")) + list(OUT.glob("light-*.md")):
        old.unlink()
    header = "Generated by `scripts/gen_rerun_queues.py`; how to run it: `docs/rerun/README.md`."
    light = {"A": [], "rest": []}
    heavy: dict[str, list[str]] = {p: [] for p in PHASES}
    for phase, cfgs in by_phase.items():
        for c in cfgs:
            ds, gen = c.split("/")[3:5]
            if ds in LIGHT:
                light["A" if phase == "A" else "rest"].append(c)
            else:
                heavy[phase].append(c)
    for name, key, what in [("light-1-generators.md", "A", "the generator alone"),
                            ("light-2-minimizers.md", "rest", "every minimizer")]:
        (OUT / name).write_text("\n".join([f"# Light arm: {what} on {', '.join(sorted(LIGHT))}", "", header, "",
                                           *[f"- [ ] {c}" for c in light[key]]]) + "\n")
    n = 0
    index = []
    for phase, cfgs in heavy.items():
        cfgs = sorted(cfgs, key=lambda c: needs_prep(*c.split("/")[3:5]))   # stable: prep cells last
        for i in range(0, len(cfgs), QUEUE_SIZE):
            n += 1
            chunk = cfgs[i:i + QUEUE_SIZE]
            combos = sorted({"/".join(c.split("/")[3:6]) for c in chunk})
            prep = any(needs_prep(*c.split("/")[3:5]) for c in chunk)
            name = f"Q{n:02d}.md"
            body = [f"# Rerun queue {n:02d} (phase {phase}): {PHASES[phase]}", "", header, "",
                    *[f"- [ ] {c}" for c in chunk]]
            (OUT / name).write_text("\n".join(body) + "\n")
            index.append((name, phase, len(chunk), prep, combos))
    lines = [f"| `light-1-generators.md` | A | {len(light['A'])} | | every `*-dummy` of {', '.join(sorted(LIGHT))} |",
             f"| `light-2-minimizers.md` | B to D | {len(light['rest'])} | | the rest of {', '.join(sorted(LIGHT))} |"]
    for name, phase, k, prep, combos in index:
        lines.append(f"| `{name}` | {phase} | {k} | {'yes' if prep else ''} | {', '.join(f'`{c}`' for c in combos)} |")
    (OUT / "queues.md").write_text(
        "# Rerun queues\n\nGenerated by `scripts/gen_rerun_queues.py`, in submission order. "
        "Phases: " + "; ".join(f"**{p}** {d}" for p, d in PHASES.items()) + ". "
        "*Prep*: the queue needs the caches built on the server first (`docs/rerun/README.md`).\n\n"
        "| Queue | Phase | Configs | Prep | Cells (dataset/generator/variant) |\n|---|---|---|---|---|\n"
        + "\n".join(lines) + "\n")
    total = sum(len(v) for v in by_phase.values())
    print(f"{total} configs ({changed} given generator_cache now, {fixed} given the v0 fix now): light arm {len(light['A'])} + "
          f"{len(light['rest'])}, {n} heavy queues: " + ", ".join(f"{p}={len(v)}" for p, v in heavy.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
