#!/usr/bin/env python3
"""The selector paper's experiment matrix: ten datasets, ten folds, DCE, four arms, and the collections
that pretrain the selector without leakage.

Arms (names in the paper in brackets), each in ``lab/config/generate_minimize/<dataset>-sel/dce/``:

    dce-lcls-v0             [bls_v0]        the thesis implementation as the paper describes it: removal
                                            levels from one edge (code fix), stop after the first outer
                                            iteration without improvement, the seven attribute strategies
    dce-lcls-sel-random     [bls_v1]        the selector skeleton with uniform proposals and no model
    dce-lcls-sel            [selector_v1]   the pretrained, frozen selector
    dce-lcls-sel-untrained  [control]       the same network and policy, untrained
    dce-lcls-sel-collect                    uniform proposals dumping every scored example (folds 1 and 2
                                            only, budget 2000), for the pretraining

Every arm takes the minimizer settings of the PROTEINS study (``gen_selector_proteins_folds.py``), the
dataset's thesis config for everything else, and the flags that config sets about how the oracle is
queried (``DATASET_FLAGS``). PROTEINS keeps its existing selector, untrained, random and
uniform-fallback configs (``proteins-sel/dce``); this script adds its v0 arm and its collections and
repoints fold 0 of the selector and uniform-fallback arms to a leakage-free checkpoint.

Pretraining without leakage: the checkpoint used on fold f must not have seen f's test instances, so each
dataset has two, trained on the collections of folds 1 and 2: ``edge_selector_sel-pre_<dataset>-f1.pt``
serves every fold but fold 1, ``-f2.pt`` serves fold 1 (``scripts/pretrain_selector_checkpoints.py``
builds whichever is missing). PROTEINS folds 1 to 9 keep ``edge_selector_v3-pre_PROTEINS.pt``, trained
on fold-0 instances, which only fold 0 evaluates.

Queues (``docs/selector/PAPER_*.md``), in the order they can run:

    PAPER_LOCAL_1.md   local: the collections (folds 1 and 2) of the datasets with local caches, and the
                       fold-0 preliminary (v0, bls_v1, control); PROTEINS collections last
    PAPER_LOCAL_2.md   local, after pretraining: selector_v1 on fold 0 of those datasets
    PAPER_Q<nn>.md     server, at most 30 configs each: S1 needs nothing (v0, bls_v1, control, folds 1-9;
                       PROTEINS v0 and uniform-fallback), S2 needs the committed checkpoints (selector_v1
                       folds 1-9, PROTEINS fold 0 of selector and uniform-fallback), S3 needs the BZR and
                       ENZYMES caches and checkpoints (everything of those two)
    PAPER_QUEUES.md    the index

    python scripts/gen_selector_paper_matrix.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TREE = REPO / "lab/config/generate_minimize"
PROTEINS = TREE / "proteins-sel/dce"
DOCS = REPO / "docs/selector"
LOCAL = ["asd", "tcr-500-28", "tcr-tco-300", "synthie", "bbbp", "aids", "colors-3"]   # caches committed
SERVER_CACHES = ["bzr", "enzymes"]                                                   # caches built on the server
DATASETS = ["proteins"] + LOCAL + SERVER_CACHES
DATASET_FLAGS = ["attributed", "recompute_features", "neigh_factor", "runtime_factor", "max_runtime", "max_neigh"]
SEVEN = ["identity", "average_smoothing", "weighted_smoothing", "laplacian_regularization", "feature_aggregation",
         "heat_kernel_diffusion", "random_walk_diffusion"]
BUDGET, COLLECT_BUDGET, COLLECT_FOLDS, FOLDS = 10000, 2000, (1, 2), range(10)
CACHE = "lab/data/cache/generator_outputs"
QUEUE_SIZE = 30


def ckpt(ds: str, fold: int) -> str:
    if ds == "proteins" and fold != 0:
        return "models/pretrained/edge_selector_v3-pre_PROTEINS.pt"
    return f"models/pretrained/edge_selector_sel-pre_{ds}-f{2 if fold == 1 else 1}.pt"


def thesis_config(ds: str, fold: int) -> dict:
    tree = PROTEINS / "dce-lcls" if ds == "proteins" else TREE / ds / "dce/dce-lcls"
    return json.load(open(tree / f"generate_minimize{fold}.jsonc"))


def set_fold(conf: dict, fold: int) -> None:
    for prop in conf["experiment"]["parameters"]["propagate"]:
        if prop["in_sections"] == ["doe-triplets/explainer"]:
            prop["params"]["fold_id"] = fold
    conf["doe-triplets"][0]["explainer"]["parameters"]["fold_id"] = fold


def write(conf: dict, ds: str, arm: str, fold: int) -> str:
    out = TREE / f"{ds}-sel/dce" / arm / f"generate_minimize{fold}.jsonc"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(conf, indent=4) + "\n")
    return str(out.relative_to(REPO))


def v0_config(ds: str, fold: int) -> dict:
    conf = thesis_config(ds, fold)
    set_fold(conf, fold)
    conf["experiment"]["scope"] = f"{ds}-sel_dce_lcls-v0"
    conf["evaluator"]["parameters"].pop("instance_ids", None)
    expl = conf["doe-triplets"][0]["explainer"]["parameters"]
    expl["generator_cache"] = CACHE
    assert expl["minimizer"]["class"].endswith("local_search.local_search.LocalSearch"), expl["minimizer"]["class"]
    par = expl["minimizer"]["parameters"]
    par.update({"seed": 0, "max_oracle_calls": BUDGET, "strict_budget": True,
                "stop_after_failed_iteration": True, "attribute_methods": list(SEVEN)})
    return conf


def skeleton_config(ds: str, arm: str, fold: int) -> dict:
    """A selector-skeleton arm on the dataset's thesis config, PROTEINS minimizer settings."""
    src_arm = "dce-lcls-sel-random" if arm == "dce-lcls-sel-collect" else arm
    src = json.load(open(PROTEINS / src_arm / "generate_minimize0.jsonc"))
    conf = thesis_config(ds, fold)
    set_fold(conf, fold)
    conf["experiment"]["scope"] = f"{ds}-sel_" + arm.replace("dce-lcls", "dce_lcls", 1)
    conf["evaluator"]["parameters"].pop("instance_ids", None)
    thesis_min = conf["doe-triplets"][0]["explainer"]["parameters"]["minimizer"]["parameters"]
    expl = conf["doe-triplets"][0]["explainer"]["parameters"]
    expl["generator_cache"] = CACHE
    expl["minimizer"] = json.loads(json.dumps(src["doe-triplets"][0]["explainer"]["parameters"]["minimizer"]))
    par = expl["minimizer"]["parameters"]
    for flag in DATASET_FLAGS:
        if flag in thesis_min:
            par[flag] = thesis_min[flag]
        else:
            par.pop(flag, None)
    par["seed"] = 0
    par["max_oracle_calls"] = COLLECT_BUDGET if arm == "dce-lcls-sel-collect" else BUDGET
    if arm == "dce-lcls-sel":
        par["init_checkpoint"] = ckpt(ds, fold)
    if arm == "dce-lcls-sel-collect":
        par["model_tag"] = "selector-collect"
        par["collect_examples"] = f"lab/output/selector_pretrain/{ds}_collect-f{fold}.pt"
    return conf


def repoint_proteins_fold0(arm: str) -> str:
    """Fold 0 of an existing PROTEINS selector arm takes the fold-1 checkpoint (its own is trained on
    fold-0 instances); the file keeps everything else."""
    path = PROTEINS / arm / "generate_minimize0.jsonc"
    conf = json.load(open(path))
    par = conf["doe-triplets"][0]["explainer"]["parameters"]["minimizer"]["parameters"]
    par["init_checkpoint"] = ckpt("proteins", 0)
    path.write_text(json.dumps(conf, indent=4) + "\n")
    return str(path.relative_to(REPO))


def main() -> int:
    local1, local2 = [], []
    s1, s2, s3 = [], [], []
    for ds in DATASETS:
        server_cached = ds in SERVER_CACHES
        v0 = {f: write(v0_config(ds, f), ds, "dce-lcls-v0", f) for f in FOLDS}
        if ds == "proteins":
            coll = {f: write(skeleton_config(ds, "dce-lcls-sel-collect", f), ds, "dce-lcls-sel-collect", f) for f in COLLECT_FOLDS}
            sel0 = repoint_proteins_fold0("dce-lcls-sel")
            fb = {f: str((PROTEINS / "dce-lcls-sel-uniform-fallback" / f"generate_minimize{f}.jsonc").relative_to(REPO)) for f in FOLDS}
            repoint_proteins_fold0("dce-lcls-sel-uniform-fallback")
            local1 += list(coll.values())                      # last in the local queue (see below)
            s1 += list(v0.values()) + [fb[f] for f in FOLDS if f != 0]
            s2 += [sel0, fb[0]]
            continue
        coll = {f: write(skeleton_config(ds, "dce-lcls-sel-collect", f), ds, "dce-lcls-sel-collect", f) for f in COLLECT_FOLDS}
        rnd = {f: write(skeleton_config(ds, "dce-lcls-sel-random", f), ds, "dce-lcls-sel-random", f) for f in FOLDS}
        unt = {f: write(skeleton_config(ds, "dce-lcls-sel-untrained", f), ds, "dce-lcls-sel-untrained", f) for f in FOLDS}
        sel = {f: write(skeleton_config(ds, "dce-lcls-sel", f), ds, "dce-lcls-sel", f) for f in FOLDS}
        if server_cached:
            s3 += list(coll.values()) + list(v0.values()) + list(rnd.values()) + list(unt.values()) + list(sel.values())
            continue
        local1 += [coll[1], coll[2], v0[0], rnd[0], unt[0]]
        local2.append(sel[0])
        s1 += [v0[f] for f in FOLDS if f] + [rnd[f] for f in FOLDS if f] + [unt[f] for f in FOLDS if f]
        s2 += [sel[f] for f in FOLDS if f]
    # PROTEINS collections go last in the local queue: the preliminary of the other datasets comes first
    prot = [c for c in local1 if "/proteins-sel/" in c]
    local1 = [c for c in local1 if c not in prot] + prot

    header = "Generated by `scripts/gen_selector_paper_matrix.py` (its docstring gives the order and the arms)."
    (DOCS / "PAPER_LOCAL_1.md").write_text(
        "# Selector paper, local queue 1: collections (folds 1 and 2) and the fold-0 preliminary (bls_v0, bls_v1, control)\n\n"
        f"{header} `python scripts/run_revision_queue.py --list docs/selector/PAPER_LOCAL_1.md --workers 1 --timeout 86400`, then "
        "`python scripts/pretrain_selector_checkpoints.py`.\n\n" + "\n".join(f"- [ ] {c}" for c in local1) + "\n")
    (DOCS / "PAPER_LOCAL_2.md").write_text(
        "# Selector paper, local queue 2: selector_v1 on fold 0 (needs the fold-1 checkpoints)\n\n"
        f"{header}\n\n" + "\n".join(f"- [ ] {c}" for c in local2) + "\n")
    for old in DOCS.glob("PAPER_Q*.md"):
        old.unlink()
    index, n = [], 0
    for phase, cfgs, needs in (("S1", s1, ""), ("S2", s2, "committed checkpoints"), ("S3", s3, "BZR and ENZYMES caches and checkpoints")):
        for i in range(0, len(cfgs), QUEUE_SIZE):
            n += 1
            chunk = cfgs[i:i + QUEUE_SIZE]
            name = f"PAPER_Q{n:02d}.md"
            (DOCS / name).write_text(f"# Selector paper, server queue {n:02d} (phase {phase}{', needs ' + needs if needs else ''})\n\n"
                                     f"{header} `launchers/slurm_submit.sh -p normal -m 10 docs/selector/{name}`.\n\n"
                                     + "\n".join(f"- [ ] {c}" for c in chunk) + "\n")
            cells = sorted({"/".join(c.split("/")[3:6]) for c in chunk})
            index.append(f"| `{name}` | {phase} | {len(chunk)} | {needs} | {', '.join(f'`{c}`' for c in cells)} |")
    (DOCS / "PAPER_QUEUES.md").write_text(
        "# Selector paper: server queues\n\nGenerated by `scripts/gen_selector_paper_matrix.py`, in submission order, one at a time "
        "(`launchers/slurm_submit.sh -p normal -m 10 <queue>`). S1 needs nothing; S2 needs the checkpoints of "
        "`scripts/pretrain_selector_checkpoints.py` committed and synced; S3 needs the BZR and ENZYMES caches (`docs/rerun/README.md`, "
        "section 2) and their checkpoints.\n\n| Queue | Phase | Configs | Needs | Cells (dataset/generator/arm) |\n|---|---|---|---|---|\n"
        + "\n".join(index) + "\n")
    print(f"local 1: {len(local1)}, local 2: {len(local2)}, server: S1 {len(s1)} + S2 {len(s2)} + S3 {len(s3)} in {n} queues")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
