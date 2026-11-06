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
* the isolated-node policy (src/utils/isolated.py): every dataset but COLORS-3, Synthie and ASD
  (``KEEP_ISOLATED``) gets DeleteIsolatedNodes first in its manipulators (a copy of the manipulator snippet
  named ``*-delete-isolated``), so the isolated nodes it has when it loads go and a node an edit leaves
  isolated is deleted;
* DownstreamGCN gets ``exact_conv_layers``, so it builds the convolutions its config asks for (v1 builds
  two whenever more are asked; the oracles are retrained in v2 anyway);
* every pipeline gets the edit cost under the node correspondence (``edit_distance.EditDistance``,
  ``StructuralEditDistance`` and ``MatchedFeatureDistance``) right after the cell-wise GED stage, which
  stays for comparison with v1, and loses the cell-wise FED stage (it needs equal sizes; MatchedFeatureDistance,
  the feature distance under the same correspondence, replaces it);
* the DCE and DCM generators get ``projection: "assignment"``, the generator contract of v2 (the full
  projection of the nearest candidate the oracle accepts, decided by the generator alone, so generation
  stays decoupled from minimization), and DCM classifies its medoids with the oracle (and subsamples its
  candidates on COLORS-3 and PROTEINS, ``DCM_PROPORTION``);
* the experiment scope and the selector's ``model_tag`` get the prefix ``v2-``, and its checkpoints and
  collections (``init_checkpoint``, ``collect_examples``) move to a ``v2/`` subfolder, so v2 never reads or
  overwrites a v1 artefact;
* a composed snippet is written to ``lab/config/v2/<its path under lab/config>`` (``lab/config/v2/<its path>``
  for the few outside it, the legacy snippets) and the reference points there;
* every plain BLS cell of the first paper with DCE or DCM also gets a BLS-F arm (``<gen>-lclsf``,
  local_search_features: edge edits and node-feature substitutions towards the seed).

Sources, regenerated from their current v1 definitions every time this runs:

* ``paper1``: the cells of ``scripts/gen_rerun_queues.py`` (thesis and paper revision), without RSGG and DBS,
  and with DFS only on the fixed-size datasets (its prior is per node index, meaningful only when index i is
  the same node in every graph; DBS, which shares that prior, waits for an adapted one);
* ``selector``: every config of the selector paper's arms in every cell of its matrix (the v1 tree).

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
FED_STAGE = "src.evaluation.future.stages.fed.FeatureEditDistance"
ISOLATED = {"class": "src.dataset.manipulators.isolated.DeleteIsolatedNodes", "parameters": {}}
# Isolated-node policy "keep" (task 0, decided 2026-11-04): datasets where isolated nodes are part of the data
# (COLORS-3 64 % of graphs, Synthie 98 %) and fixed-node ones (ASD). Every other dataset deletes a node an
# edit leaves isolated, and loses the isolated nodes it has when it loads (salt ions in molecules).
KEEP_ISOLATED = ("colors-3", "synthie", "asd")
DCM = "src.explainer.future.search.dcm.DCM"
DCM_PROPORTION = {"colors-3": 0.2, "proteins": 0.5}    # 10 500 graphs; graphs of up to 620 nodes
BLS = "src.explainer.future.metaheuristic.local_search.local_search.LocalSearch"
BLSF = "src.explainer.future.metaheuristic.local_search.local_search_features.LocalSearch"
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
        self.snippets: dict[str, str] = {}      # (v1 snippet path, policy) -> v2 snippet path, written once each
        self.policy = "keep"                    # isolated-node policy of the config being transformed
        self.cell = ""                          # its cell (dataset tree name)

    def node(self, value):
        if isinstance(value, list):
            out = []
            for v in value:
                if isinstance(v, dict) and v.get("class") in (PADDER, FED_STAGE):
                    continue   # no padding; the cell-wise FED needs equal sizes, MatchedFeatureDistance replaces it
                out.append(self.node(v))
                if isinstance(v, dict) and v.get("class") == GED_STAGE:
                    out += [{"class": cls, "parameters": {}} for cls in EDIT_STAGES]
            return out
        if not isinstance(value, dict):
            return value
        out = {}
        for key, item in value.items():
            if key.startswith("compose") and isinstance(item, str) and item.endswith((".json", ".jsonc")):
                out[key] = self.snippet(item, isolated=key == "compose_man" and self.policy == "delete")
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
            out.setdefault("parameters", {})["projection"] = "assignment"
            if cls == DCM:
                out["parameters"]["classify_with"] = "oracle"
                # The v2 medoids cost one Hungarian assignment per pair of graphs of different classes: on
                # the largest datasets the candidates are subsampled (deterministic, random_seed).
                if self.cell.startswith(tuple(DCM_PROPORTION)):
                    out["parameters"]["proportion"] = next(v for k, v in DCM_PROPORTION.items() if self.cell.startswith(k))
        return out

    def snippet(self, rel: str, isolated: bool = False) -> str:
        """The v2 copy of a snippet; a manipulator snippet of a dataset with the policy "delete" gets its own
        copy (name suffix -delete-isolated) with DeleteIsolatedNodes first."""
        key = (rel, isolated)
        if key not in self.snippets:
            target = v2_path(rel)
            if isolated:
                target = re.sub(r"\.(jsonc?)$", r"-delete-isolated.\1", target)
            self.snippets[key] = target
            source = REPO / rel.removeprefix("./")
            data = self.node(read_jsonc(source))
            if isolated:
                data["manipulators"] = [dict(ISOLATED)] + data.get("manipulators", [])
            out = REPO / target.removeprefix("./")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(data, indent=4) + "\n")
        return self.snippets[key]

    def config(self, rel: str, cell: str) -> tuple[str, dict]:
        self.policy = "keep" if cell.startswith(KEEP_ISOLATED) else "delete"
        self.cell = cell
        data = self.node(read_jsonc(REPO / rel))
        data["experiment"]["scope"] = PREFIX + data["experiment"]["scope"]
        if self.policy == "delete" and "compose_man" not in json.dumps(data):
            raise ValueError(f"{rel}: no manipulator snippet to carry the isolated-node policy")
        return v2_path(rel), data


def bls_f(rel: str, data: dict) -> tuple[str, dict]:
    """The BLS-F arm of a BLS config: same cell, minimizer local_search_features (edge edits and node-feature
    substitutions towards the seed), variant <gen>-lclsf, scope suffix lclsf."""
    data = json.loads(json.dumps(data))
    minimizer = data["doe-triplets"][0]["explainer"]["parameters"]["minimizer"]
    assert minimizer["class"] == BLS, rel
    minimizer["class"] = BLSF
    minimizer["parameters"]["attributed"] = False
    data["experiment"]["scope"] = re.sub(r"_lcls$", "_lclsf", data["experiment"]["scope"])
    parts = Path(rel).parts
    variant = parts[-2]
    return str(Path(*parts[:-2], variant.replace("-lcls", "-lclsf"), parts[-1])), data


def paper1_configs() -> list[str]:
    out = []
    for ds, gen, var in sorted(cells()):
        if gen.startswith("rsgg"):
            continue
        if var.endswith("-dbs"):
            continue   # DBS stays out of v2 for now (decided 2026-11-04): its prior is to be adapted first
        if gen == "dfs" and ds not in FIXED_SIZE_DATASETS:
            continue
        out += sorted(str(p.relative_to(REPO)) for p in (REPO / TREE / ds / gen / var).glob("generate_minimize*.jsonc"))
    return out


def selector_configs() -> list[str]:
    """Every config of the selector paper's arms in every cell of its matrix, from the v1 tree. Not from its batches
    (docs/selector/PAPER_B*.md): those leave out what already had v1 results, which v2 has to run too."""
    from gen_selector_paper_matrix import cells as selector_cells
    arms = ("dce-lcls-v0", "dce-lcls-v1", "dce-lcls-sel-v1", "dce-lcls-sel-v1-untrained", "dce-lcls-collect2")
    out = []
    for cell in selector_cells():
        for arm in arms:
            out += sorted(str(p.relative_to(REPO)) for p in (REPO / TREE / f"{cell}-sel" / "dce" / arm).glob("generate_minimize*.jsonc"))
    return out


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
            cell = Path(rel).parts[3]
            target, data = transformer.config(rel, cell)
            if target in written:
                continue
            out = REPO / target
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(f"// Protocol v2, generated by scripts/gen_v2_configs.py from {rel}: do not edit.\n"
                           + json.dumps(data, indent=4) + "\n")
            written[target] = family
            parts = Path(target).parts
            rows.append((family, parts[4], parts[5], parts[6]))
            # BLS-F next to every plain BLS cell of the first paper with DCE or DCM (2026-11-04)
            if family == "paper1" and re.search(r"/(dce|dcm)/(dce|dcm)-lcls/", rel):
                f_target, f_data = bls_f(target, data)
                f_out = REPO / f_target
                f_out.parent.mkdir(parents=True, exist_ok=True)
                f_out.write_text(f"// Protocol v2 BLS-F arm, generated by scripts/gen_v2_configs.py from {rel}: do not edit.\n"
                                 + json.dumps(f_data, indent=4) + "\n")
                written[f_target] = family
                fp = Path(f_target).parts
                rows.append((family, fp[4], fp[5], fp[6]))
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
