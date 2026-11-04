#!/usr/bin/env python3
"""The protocol v2 configs: the v1 configs of the first paper and of the selector paper, without padding.

Protocol v2 (task 0 of the research plan, decided on 2026-11-03; ``docs/v2/README.md``) runs every graph at
its own size: no AdjacencyMatrixPadder, so every node of a graph is real, the minimizers search only real
node pairs and the oracles never see padding. v1 stays as it is, padding included, for the explainers that
need a fixed size (RSGG, CLEAR, CF2, the GANs) and for reproducing the published results. Nothing of v1 is
modified: every v2 config and snippet is a transformed copy under ``lab/config/v2/``.

The transformation, applied to a config and, recursively, to every snippet it composes (any key starting
with ``compose`` whose value is a file, also inside ``propagate`` blocks):

* AdjacencyMatrixPadder leaves every ``manipulators`` list;
* DownstreamGCN gets ``exact_conv_layers``, so it builds the convolutions its config asks for (v1 builds
  two whenever more are asked; the oracles are retrained in v2 anyway);
* every pipeline gets the edit cost under the node correspondence (``edit_distance.EditDistance``,
  ``StructuralEditDistance`` and ``MatchedFeatureDistance``) right after the cell-wise GED stage, which
  stays for comparison with v1;
* the DCE and DCM generators get ``projection: "index"``, the generator contract of v2: the nearest
  candidate whose projection onto the input's nodes is a counterfactual (implemented in step 3 of the plan;
  until then the v2 configs with DCE or DCM do not run);
* the experiment scope and the selector's ``model_tag`` get the prefix ``v2-``, and its checkpoints and
  collections (``init_checkpoint``, ``collect_examples``) move to a ``v2/`` subfolder, so v2 never reads or
  overwrites a v1 artefact;
* a composed snippet is written to ``lab/config/v2/<its path under lab/config>`` (``lab/config/v2/<its path>``
  for the few outside it, the legacy snippets) and the reference points there.

Sources, regenerated from their current v1 definitions every time this runs:

* ``paper1``: the cells of ``scripts/gen_rerun_queues.py`` (thesis and paper revision), without RSGG, and
  with DFS and DBS only on the fixed-size datasets (their priors are per node index, meaningful only when
  index i is the same node in every graph);
* ``selector``: every config of the selector paper's batches (``docs/selector/PAPER_B*.md``).

Configs that use a fixed-size explainer are skipped and listed. The run order is decided later; this only
writes the configs and ``docs/v2/CONFIGS.md``, the index.

    python scripts/gen_v2_configs.py
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from gen_rerun_queues import TREE, cells  # noqa: E402

V2 = REPO / "lab/config/v2"
PREFIX = "v2-"
PADDER = "src.dataset.manipulators.padding.AdjacencyMatrixPadder"
GCN = "src.oracle.nn.gcn.DownstreamGCN"
GED_STAGE = "src.evaluation.future.stages.ged.GraphEditDistance"
EDIT_STAGES = [f"src.evaluation.future.stages.edit_distance.{name}"
               for name in ("EditDistance", "StructuralEditDistance", "MatchedFeatureDistance")]
PROJECTED = ("src.explainer.future.search.dces.DCESExplainer", "src.explainer.future.search.dcm.DCM")
FIXED_SIZE_DATASETS = {"asd", "tcr-500-28", "tcr-tco-300"}
# Explainers that need every graph of a dataset at one size: v1 only (task 0, section 4.5).
FIXED_SIZE = re.compile(r"rsgg|clear|cf2|\.gans?\.|ddbs", re.IGNORECASE)
MOVED = {"init_checkpoint", "collect_examples"}
COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)
COMMENT_LINE = re.compile(r"(?m)^\s*//.*$")


def read_jsonc(path: Path):
    return json.loads(COMMENT_LINE.sub("", COMMENT_BLOCK.sub("", path.read_text())))


def v2_path(rel: str) -> str:
    """lab/config/<x> -> lab/config/v2/<x>, any other <x> (legacy snippets) -> lab/config/v2/<x>, keeping a
    leading "./" if there was one."""
    lead = "./" if rel.startswith("./") else ""
    rel = rel[len(lead):]
    assert not rel.startswith("lab/config/v2/"), rel
    return lead + "lab/config/v2/" + rel.removeprefix("lab/config/")


def moved(path: str) -> str:
    p = Path(path)
    return str(p.parent / "v2" / p.name)


class Transformer:
    def __init__(self):
        self.snippets: dict[str, str] = {}      # v1 snippet path -> v2 snippet path, written once each

    def node(self, value):
        if isinstance(value, list):
            out = []
            for v in value:
                if isinstance(v, dict) and v.get("class") == PADDER:
                    continue
                out.append(self.node(v))
                if isinstance(v, dict) and v.get("class") == GED_STAGE:
                    out += [{"class": cls, "parameters": {}} for cls in EDIT_STAGES]
            return out
        if not isinstance(value, dict):
            return value
        out = {}
        for key, item in value.items():
            if key.startswith("compose") and isinstance(item, str) and item.endswith((".json", ".jsonc")):
                out[key] = self.snippet(item)
            elif key == "model_tag" and isinstance(item, str):
                out[key] = PREFIX + item
            elif key in MOVED and isinstance(item, str):
                out[key] = moved(item)
            else:
                out[key] = self.node(item)
        cls = out.get("class")
        if cls == GCN:
            out.setdefault("parameters", {})["exact_conv_layers"] = True
        elif cls in PROJECTED:
            out.setdefault("parameters", {})["projection"] = "index"
        return out

    def snippet(self, rel: str) -> str:
        if rel not in self.snippets:
            target = v2_path(rel)
            self.snippets[rel] = target
            source = REPO / rel.removeprefix("./")
            data = self.node(read_jsonc(source))
            out = REPO / target.removeprefix("./")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(data, indent=4) + "\n")
        return self.snippets[rel]

    def config(self, rel: str) -> tuple[str, dict]:
        data = self.node(read_jsonc(REPO / rel))
        data["experiment"]["scope"] = PREFIX + data["experiment"]["scope"]
        return v2_path(rel), data


def paper1_configs() -> list[str]:
    out = []
    for ds, gen, var in sorted(cells()):
        if gen.startswith("rsgg"):
            continue
        if (gen == "dfs" or var.endswith("-dbs")) and ds not in FIXED_SIZE_DATASETS:
            continue
        out += sorted(str(p.relative_to(REPO)) for p in (REPO / TREE / ds / gen / var).glob("generate_minimize*.jsonc"))
    return out


def selector_configs() -> list[str]:
    out = []
    for queue in sorted((REPO / "docs/selector").glob("PAPER_B*.md")):
        out += [m.group(1) for m in re.finditer(r"^- \[[ x]\] (\S+)$", queue.read_text(), re.MULTILINE)]
    return sorted(set(out))


def main() -> int:
    if V2.exists():
        shutil.rmtree(V2)
    transformer = Transformer()
    families = {"paper1": paper1_configs(), "selector": selector_configs()}
    written, skipped, rows = {}, Counter(), []
    for family, configs in families.items():
        for rel in configs:
            raw = (REPO / rel).read_text()
            if FIXED_SIZE.search(raw):
                skipped[family] += 1
                continue
            target, data = transformer.config(rel)
            if target in written:
                continue
            out = REPO / target
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(f"// Protocol v2, generated by scripts/gen_v2_configs.py from {rel}: do not edit.\n"
                           + json.dumps(data, indent=4) + "\n")
            written[target] = family
            parts = Path(target).parts
            rows.append((family, parts[4], parts[5], parts[6]))
    index = Counter(rows)
    lines = ["# Protocol v2 configs", "",
             "Generated by `scripts/gen_v2_configs.py` (its docstring gives the transformation). Do not edit the files "
             "under `lab/config/v2/`: change the v1 config or the script and regenerate. The run order is not decided "
             "yet; the DCE and DCM configs need the generator contract (step 3 of the plan) before they run.", "",
             f"{len(written)} configs ({', '.join(f'{f} {sum(1 for v in written.values() if v == f)}' for f in families)}), "
             f"{len(transformer.snippets)} transformed snippets. Skipped for a fixed-size explainer: "
             f"{dict(skipped) or 'none'}.", "",
             "| Family | Dataset | Generator | Variant | Folds |", "|---|---|---|---|---|"]
    lines += [f"| {f} | {d} | {g} | {v} | {n} |" for (f, d, g, v), n in sorted(index.items())]
    (REPO / "docs/v2").mkdir(parents=True, exist_ok=True)
    (REPO / "docs/v2/CONFIGS.md").write_text("\n".join(lines) + "\n")
    print(f"{len(written)} configs, {len(transformer.snippets)} snippets, skipped {dict(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
