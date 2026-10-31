#!/usr/bin/env python3
"""The selector paper's experiment matrix: eleven datasets, ten folds, DCE, four arms, and the collections
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

Subsets (``SUBSETS``): a pseudo-dataset such as ``colors-3-s20`` is a large dataset evaluated on a fixed
random sample of each fold's test instances (the evaluator's ``instance_ids``, hash-neutral), with its own
scope, config tree (``colors-3-s20-sel``), collections and checkpoints. They exist for the local
preliminary only and never enter the server queues.

Queues (``docs/selector/PAPER_*.md``), in the order they can run:

    PAPER_LOCAL_1.md   local, the fold-0 preliminary of ``PRELIM``: the collections of folds 1 and 2 and
                       bls_v1 on fold 0 (``python scripts/run_revision_queue.py --list ... --workers 1
                       --timeout 86400``), then ``scripts/pretrain_selector_checkpoints.py``
    PAPER_LOCAL_2.md   local, after the pretraining: selector_v1 on fold 0 of ``PRELIM``
    PAPER_Q<nn>.md     server, at most 30 configs each, every dataset of ``DATASETS`` on every fold (what
                       the local preliminary already produced is skipped by ``slurm_submit.sh`` once its
                       results are committed): S1 needs nothing (collections not run locally, v0, bls_v1,
                       control, PROTEINS v0 and uniform-fallback), S2 needs the committed checkpoints
                       (selector_v1, PROTEINS fold 0 of selector and uniform-fallback), S3 needs the
                       committed caches of ``PENDING_CACHES`` (everything of those datasets).
                       Collections run on the server come back with ``make server-import``
                       (``lab/output/selector_pretrain``) for the pretraining, which runs here
    PAPER_QUEUES.md    the index
    PAPER_SCREEN.md    local, the screening of candidate datasets (``SCREEN``): bls_v0 and bls_v1 on a fixed
                       sample of fold 0, on a copy of the dataset without its largest graphs (``max_nodes``
                       of the TU generator, so that the global padding fits in memory). Screening only: a
                       dataset that passes enters the matrix complete

    python scripts/gen_selector_paper_matrix.py
"""
from __future__ import annotations

import json
import pickle
import re
from collections import OrderedDict
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TREE = REPO / "lab/config/generate_minimize"
PROTEINS = TREE / "proteins-sel/dce"
DOCS = REPO / "docs/selector"
# The datasets of the first paper. Tree-Cycles (tcr-500-28, tcr-tco-300) is stale since 2026-10-30 (user's decision:
# every method reaches the minimum, it tells nothing); its configs stay.
# ASD is stale too since 2026-10-30 (its oracle is a fixed rule: neither the margin nor the robust retraining apply).
DATASETS = ["proteins", "synthie", "bbbp", "aids", "colors-3", "bzr", "enzymes"]
PENDING_CACHES = ["bzr", "enzymes"]      # dataset and oracle caches not committed yet (bzr is built by the local preliminary, enzymes on the server)
PRELIM = ["aids", "bbbp", "synthie", "bzr", "colors-3-s20"]   # the fold-0 preliminary on this machine, in queue order
ADHD_EXPS, ADHD_CACHES = [], {}      # filled once the ADHD oracle exists
LOCAL_TCR = []   # Tree-Cycles is stale (see DATASETS); it was ["tcr-500-28", "tcr-tco-300"], fold 0 again after the generator fix
LOCAL_REDO = ["bbbp"]   # rerun locally after the BBBP order was fixed (2026-10-30): collections and bls_v1 (PAPER_LOCAL_3), selector_v1 (PAPER_LOCAL_4)
SUBSETS = {"colors-3-s20": ("colors-3", {0: 20, 1: 40, 2: 40}),           # pseudo-dataset: (base dataset, sampled test instances per fold)
           "colors-3-m90-s20": ("colors-3-m90", {0: 20, 1: 40, 2: 40}),
           "colors-3-rob-s20": ("colors-3-rob", {0: 20, 1: 40, 2: 40})}
# The three oracles of the paper's tables (2026-10-30): the first paper's (the trees above), the same with a
# confidence margin (MarginOracle, <dataset>-m90) and one retrained with random edge perturbations
# (OracleTorch edge_perturbation, <dataset>-rob; not for asd, whose oracle is a fixed rule).
MARGIN = 0.9
ROBUST = {"drop": 0.1, "add": 0.1}
VARIANTS = {"m90": list(DATASETS), "rob": list(DATASETS)}
# tri, the triangle-threshold synthetic, has its own table (2026-10-30): several configurations, each with the
# exact rule as reference and the three oracles on a GCN (with each node's triangle count as a feature):
# <tri>-gcn, <tri>-gcn-m90, <tri>-gcn-rob. All of one configuration share the dataset cache and folds.
TRI = {"tri-40": "TRI-500-40", "tri-40n": "TRI-500-40n", "tri-80": "TRI-500-80"}
TRI_EXPS = [e for t in TRI for e in (t, f"{t}-gcn", f"{t}-gcn-m90", f"{t}-gcn-rob")]
SONDEO_LOCAL = ["bbbp"]   # its original-oracle fold 0 is being rerun locally (PAPER_LOCAL_3/4), left out of the server sondeo
# Screening of candidate datasets (2026-10-30): (config tree of the dataset, sampled fold-0 test instances)
# DD and REDDIT-BINARY were dropped on 2026-10-30 until the padding is per pair (task A3 of the PhD plan)
# IMDB uses its own tree (imdb-deg): the thesis IMDB oracle predicts one class for all 1000 graphs.
# Harder experiments (2026-10-30): the confidence-margin oracle (src/oracle/margin.py) on PROTEINS and bbbp
# (aids left out: its oracle is saturated, the minimal counterfactuals already have a confidence >= 0.9 in
# 67 % of the instances), the triangle-threshold synthetic (tri-40) with and without margin, and ADHD.
NEW = ["proteins-m70", "proteins-m90", "bbbp-m70", "bbbp-m90"] + ADHD_EXPS   # the local screening of 2026-10-30 (PAPER_SCREEN.md)
SCREEN = {"imdb-s30": ("imdb-deg", 30), "mutagenicity-s30": ("mutagenicity-c100", 30),
          **{f"{e}-s20": (e, 20) for e in NEW}}
# (the tri-40, tri-40-m70 and tri-40-m90 screening of 2026-10-30 ran on tri-40's first tree, without node
# features; its configs and results stay as they are, outside the generator)
CACHE_PREFIX = {"colors-3": "COLORS-3", "imdb-deg": "IMDB-BINARY-97c8af347f83bb14228bd7f27567e0c4",
                "mutagenicity-c100": "Mutagenicity", "colors-3-m90": "COLORS-3", "colors-3-rob": "COLORS-3",
                **{f"proteins-m{t}": "PROTEINS-084c6a1896e8a96ebacf7ff4868f1197" for t in (70, 90)},
                **{f"bbbp-m{t}": "BBBP-c0cc58bdf4acf93936322304bf6b669e" for t in (70, 90)},
                **{e: "TriangleThreshold-cd6e25aea0d32bb9b91d5eaeeb5b4122" for e in ("tri-40", "tri-40-m70", "tri-40-m90")},
                **ADHD_CACHES}   # dataset cache file prefix (or full name) of each config tree
N_FOLDS = {}       # datasets with fewer than 10 folds (StratifiedKFold is capped by the smallest class); none since cuneiform left the matrix (2026-10-30, its classes are told apart by node positions: DCE finds a same-edge-set counterfactual, GED 0)
DATASET_FLAGS = ["attributed", "recompute_features", "neigh_factor", "runtime_factor", "max_runtime", "max_neigh"]
SEVEN = ["identity", "average_smoothing", "weighted_smoothing", "laplacian_regularization", "feature_aggregation",
         "heat_kernel_diffusion", "random_walk_diffusion"]
BUDGET, COLLECT_BUDGET, COLLECT_FOLDS = 10000, 2000, (1, 2)
CACHE = "lab/data/cache/generator_outputs"
QUEUE_SIZE = 30


def base(ds: str) -> str:
    if ds in SCREEN:
        return SCREEN[ds][0]
    return SUBSETS[ds][0] if ds in SUBSETS else ds


def folds(ds: str) -> range:
    return range(N_FOLDS.get(base(ds), 10))


def ckpt(ds: str, fold: int) -> str:
    if ds == "proteins" and fold != 0:
        return "models/pretrained/edge_selector_v3-pre_PROTEINS.pt"
    return f"models/pretrained/edge_selector_sel-pre_{ds}-f{2 if fold == 1 else 1}.pt"


def thesis_config(ds: str, fold: int) -> dict:
    ds = base(ds)
    tree = PROTEINS / "dce-lcls" if ds == "proteins" else TREE / ds / "dce/dce-lcls"
    return json.load(open(tree / f"generate_minimize{fold}.jsonc"))


def subset_ids(ds: str, fold: int) -> list[int]:
    """A fixed random sample of the fold's test instances, from the committed dataset cache (TU datasets
    number their instances by position, so split indices are instance ids)."""
    name, sizes = SUBSETS[ds] if ds in SUBSETS else (SCREEN[ds][0], {0: SCREEN[ds][1]})
    caches = list((REPO / "lab/data/cache/datasets").glob(f"{CACHE_PREFIX[name]}*"))
    assert len(caches) == 1, f"{name}: expected one committed dataset cache, found {caches}"
    splits = pickle.load(open(caches[0], "rb"))["splits"]
    return sorted(random.Random(1000 + fold).sample(sorted(splits[fold]["test"]), sizes[fold]))


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
    if ds in SUBSETS or ds in SCREEN:
        conf["evaluator"]["parameters"]["instance_ids"] = subset_ids(ds, fold)
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


def template_dir(ds: str) -> Path:
    return PROTEINS / "dce-lcls" if ds == "proteins" else TREE / ds / "dce/dce-lcls"


def make_tri_templates() -> None:
    """The tri trees, all ten folds, from Tree-Cycles' (no padding): the exact rule (<tri>) and the GCN
    (<tri>-gcn), both with the node-triangle feature so that they share the dataset cache."""
    for name, pair in TRI.items():
        for tree, snip in ((name, f"{pair}_Custom"), (f"{name}-gcn", f"{pair}_GCN")):
            for f in range(10):
                t = (TREE / "tcr-500-28/dce/dce-lcls" / f"generate_minimize{f}.jsonc").read_text()
                t = t.replace("do-pairs/TCR-500-28-0.25_Custom.json", f"do-pairs/{snip}.json")
                t = t.replace("lab/config/snippets/datasets/empty.json", "lab/config/snippets/datasets/triangles.json")
                conf = json.loads(t)
                conf["experiment"]["scope"] = f"{tree}_dce_lcls"
                conf["doe-triplets"][0]["explainer"]["parameters"]["minimizer"]["parameters"]["recompute_features"] = True
                out = TREE / tree / "dce/dce-lcls" / f"generate_minimize{f}.jsonc"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(conf, indent=4) + "\n")


def make_variant(ds: str, v: str) -> str:
    """The thesis-style tree <ds>-<v> (every fold) with the dataset-oracle pair of <ds> turned into the
    variant's oracle: MarginOracle around the original one (loaded from its cache) or the same oracle
    retrained with edge perturbations. Returns the variant's name."""
    name = f"{ds}-{v}"
    for f in range(10):
        src = template_dir(ds) / f"generate_minimize{f}.jsonc"
        if not src.exists():
            continue
        text = src.read_text()
        pair = re.search(r'do-pairs/([^"]+)\.json', text).group(1)
        snip_path = REPO / f"lab/config/snippets/do-pairs/{pair}_{v}.json"
        if not snip_path.exists() or f == 0:
            orig = json.loads((REPO / f"lab/config/snippets/do-pairs/{pair}.json").read_text(), object_pairs_hook=OrderedDict)
            oracle = json.loads(json.dumps(orig["oracle"]), object_pairs_hook=OrderedDict)
            if v.startswith("m"):
                oracle["parameters"]["fold_id"] = -1
                oracle["parameters"]["retrain"] = False
                oracle = OrderedDict([("class", "src.oracle.margin.MarginOracle"),
                                      ("parameters", OrderedDict([("margin", MARGIN), ("base_oracle", oracle)]))])
            else:
                assert oracle["class"].endswith("OracleTorch"), (ds, oracle["class"])
                oracle["parameters"]["edge_perturbation"] = dict(ROBUST)
            snip_path.write_text(json.dumps(OrderedDict(dataset=orig["dataset"], oracle=oracle), indent=4) + "\n")
        conf = json.loads(text.replace(f"do-pairs/{pair}.json", f"do-pairs/{pair}_{v}.json"))
        conf["experiment"]["scope"] = f"{name}_dce_lcls"
        out = TREE / name / "dce/dce-lcls" / f"generate_minimize{f}.jsonc"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(conf, indent=4) + "\n")
    return name


def main() -> int:
    local1, local2, local_tcr, local_redo = {}, {}, [], []
    s1, s2, s3 = [], [], []
    sondeo_a, sondeo_b, sondeo_ds = [], [], []
    make_tri_templates()
    for v, dss in VARIANTS.items():
        for ds in dss:
            make_variant(ds, v)
    for t in TRI:
        for v in VARIANTS:
            make_variant(f"{t}-gcn", v)
    for ds in DATASETS:
        v0 = {f: write(v0_config(ds, f), ds, "dce-lcls-v0", f) for f in folds(ds)}
        coll = {f: write(skeleton_config(ds, "dce-lcls-sel-collect", f), ds, "dce-lcls-sel-collect", f) for f in COLLECT_FOLDS}
        if ds == "proteins":
            sel0 = repoint_proteins_fold0("dce-lcls-sel")
            fb = {f: str((PROTEINS / "dce-lcls-sel-uniform-fallback" / f"generate_minimize{f}.jsonc").relative_to(REPO)) for f in folds(ds)}
            repoint_proteins_fold0("dce-lcls-sel-uniform-fallback")
            s1 += list(coll.values()) + list(v0.values()) + [fb[f] for f in folds(ds) if f != 0]
            s2 += [sel0, fb[0]]
            sondeo_a += [coll[1], coll[2], v0[0], str((PROTEINS / "dce-lcls-sel-random/generate_minimize0.jsonc").relative_to(REPO))]
            sondeo_b.append(sel0)
            sondeo_ds.append(ds)
            continue
        rnd = {f: write(skeleton_config(ds, "dce-lcls-sel-random", f), ds, "dce-lcls-sel-random", f) for f in folds(ds)}
        unt = {f: write(skeleton_config(ds, "dce-lcls-sel-untrained", f), ds, "dce-lcls-sel-untrained", f) for f in folds(ds)}
        sel = {f: write(skeleton_config(ds, "dce-lcls-sel", f), ds, "dce-lcls-sel", f) for f in folds(ds)}
        if ds not in SONDEO_LOCAL and ds != "colors-3":
            sondeo_a += [coll[1], coll[2], v0[0], rnd[0]]
            sondeo_b.append(sel[0])
            sondeo_ds.append(ds)
        if ds in PRELIM:
            local1[ds] = [coll[1], coll[2], rnd[0]]
            local2[ds] = [sel[0]]
        if ds in LOCAL_TCR:
            local_tcr += [coll[1], coll[2], v0[0], rnd[0], unt[0]]
        if ds in LOCAL_REDO:
            local_tcr += [coll[1], coll[2], rnd[0]]
            local_redo.append(sel[0])
        server_coll = [] if ds in PRELIM else list(coll.values())
        if ds in PENDING_CACHES:
            s3 += server_coll + list(v0.values()) + list(rnd.values()) + list(unt.values()) + list(sel.values())
        else:
            s1 += server_coll + list(v0.values()) + list(rnd.values()) + list(unt.values())
            s2 += list(sel.values())
    for ds in SUBSETS:
        coll = {f: write(skeleton_config(ds, "dce-lcls-sel-collect", f), ds, "dce-lcls-sel-collect", f) for f in COLLECT_FOLDS}
        local1[ds] = [coll[1], coll[2], write(skeleton_config(ds, "dce-lcls-sel-random", 0), ds, "dce-lcls-sel-random", 0)]
        local2[ds] = [write(skeleton_config(ds, "dce-lcls-sel", 0), ds, "dce-lcls-sel", 0)]
    local1 = [c for ds in PRELIM for c in local1[ds]]
    local2 = [c for ds in PRELIM for c in local2[ds]]

    header = "Generated by `scripts/gen_selector_paper_matrix.py` (its docstring gives the order and the arms)."
    prelim = ", ".join(PRELIM)
    (DOCS / "PAPER_LOCAL_1.md").write_text(
        f"# Selector paper, local queue 1: the fold-0 preliminary of {prelim}: collections (folds 1 and 2) and bls_v1\n\n"
        f"{header} `python scripts/run_revision_queue.py --list docs/selector/PAPER_LOCAL_1.md --workers 1 --timeout 86400`, then "
        "`python scripts/pretrain_selector_checkpoints.py`.\n\n" + "\n".join(f"- [ ] {c}" for c in local1) + "\n")
    (DOCS / "PAPER_LOCAL_2.md").write_text(
        f"# Selector paper, local queue 2: selector_v1 on fold 0 of {prelim} (needs the fold-1 checkpoints)\n\n"
        f"{header}\n\n" + "\n".join(f"- [ ] {c}" for c in local2) + "\n")
    screen = []
    for ds in SCREEN:
        if not list((REPO / "lab/data/cache/datasets").glob(f"{CACHE_PREFIX[base(ds)]}*")):
            print(f"screening: {ds} skipped, no dataset cache yet (tools/warm_caches.py on {base(ds)}/dce/dce-lcls)")
            continue
        conf = v0_config(ds, 0)
        conf["evaluator"]["parameters"]["instance_ids"] = subset_ids(ds, 0)
        screen += [write(conf, ds, "dce-lcls-v0", 0), write(skeleton_config(ds, "dce-lcls-sel-random", 0), ds, "dce-lcls-sel-random", 0)]
    (DOCS / "PAPER_SCREEN.md").write_text(
        "# Selector paper: screening of candidate datasets (bls_v0 and bls_v1 on 30 fixed fold-0 instances)\n\n"
        f"{header} Datasets without their largest graphs (TU generator ``max_nodes``), for the screening only. "
        "`python scripts/run_revision_queue.py --list docs/selector/PAPER_SCREEN.md --workers 1 --timeout 86400`.\n\n"
        + "\n".join(f"- [ ] {c}" for c in screen) + "\n")
    # Sondeo (2026-10-30): fold 0 of every dataset with every oracle (colors-3 on its 20-instance subset),
    # collections of folds 1 and 2, bls_v0, bls_v1 and selector_v1, to design selector_v2 on all of it.
    def subset_v0(name):
        conf = v0_config(name, 0)
        if name in SUBSETS:
            conf["evaluator"]["parameters"]["instance_ids"] = subset_ids(name, 0)
        return write(conf, name, "dce-lcls-v0", 0)
    sondeo_a.append(subset_v0("colors-3-s20"))   # its collections, bls_v1 and selector_v1 are the local preliminary's
    for v, dss in VARIANTS.items():
        for ds in dss:
            name = f"colors-3-{v}-s20" if ds == "colors-3" else f"{ds}-{v}"
            sondeo_a += [write(skeleton_config(name, "dce-lcls-sel-collect", f), name, "dce-lcls-sel-collect", f) for f in COLLECT_FOLDS]
            sondeo_a += [subset_v0(name), write(skeleton_config(name, "dce-lcls-sel-random", 0), name, "dce-lcls-sel-random", 0)]
            sondeo_b.append(write(skeleton_config(name, "dce-lcls-sel", 0), name, "dce-lcls-sel", 0))
            sondeo_ds.append(name)
    for name in TRI_EXPS:
        sondeo_a += [write(skeleton_config(name, "dce-lcls-sel-collect", f), name, "dce-lcls-sel-collect", f) for f in COLLECT_FOLDS]
        sondeo_a += [subset_v0(name), write(skeleton_config(name, "dce-lcls-sel-random", 0), name, "dce-lcls-sel-random", 0)]
        sondeo_b.append(write(skeleton_config(name, "dce-lcls-sel", 0), name, "dce-lcls-sel", 0))
        sondeo_ds.append(name)
    warm = ["enzymes"] + [f"{ds}-rob" for ds in VARIANTS["rob"] if ds != "bzr"]   # bzr-rob and the tri oracles are committed
    steps = (
        "Rule: on the server only Slurm jobs compute (nothing heavy on the login node), and at most 10 jobs at a time. "
        "Each step starts once `squeue -u $USER` shows nothing left of the previous one.\n\n"
        "**Step 0. Sync** (only git and the pinned environment):\n\n```bash\ncd ~/projects/GRETEL\nlaunchers/server_sync.sh\n"
        "conda activate GRTL\n```\n\n"
        f"**Step 1. Caches: ENZYMES' dataset and oracle, and the robust oracles** ({len(warm)} jobs):\n\n```bash\nG=lab/config/generate_minimize\n"
        "for t in " + " ".join(warm) + "; do sbatch -p normal launchers/slurm_warm.sh $G/$t/dce/dce-lcls/generate_minimize0.jsonc; done\n"
        "squeue -u $USER\ngrep -L \"exit status 0\" output/slurm/warm_<job>.out     # for each job number sbatch printed: a file listed here failed\n```\n\n"
        f"**Step 2. Queue A**: collections of folds 1 and 2, bls_v0 and bls_v1 on fold 0 ({len(sondeo_a)} configs, 10 at a time):\n\n"
        "```bash\nlaunchers/slurm_submit.sh -p normal -m 10 docs/selector/PAPER_SONDEO_A.md\nsqueue -u $USER\n"
        "grep -lE \"exit status [1-9]|Traceback\" output/logs/*/*/*_<job>_*.out     # <job>: the array job number slurm_submit printed\n```\n\n"
        "A queue that ends with failures can be submitted again as it is: finished configs are skipped.\n\n"
        "**Step 3. Pretraining** of the selector checkpoints from those collections (1 job):\n\n"
        "```bash\nsbatch -p normal launchers/slurm_pretrain.sh " + " ".join(sondeo_ds) + "\nsqueue -u $USER\ntail output/slurm/pretrain_<job>.out     # ends with: built N checkpoint(s); collections still missing: none\n```\n\n"
        f"**Step 4. Queue B**: selector_v1 on fold 0 ({len(sondeo_b)} configs):\n\n"
        "```bash\nlaunchers/slurm_submit.sh -p normal -m 10 docs/selector/PAPER_SONDEO_B.md\nsqueue -u $USER\n```\n\n"
        "**Step 5. Back home** (local machine): `make server-import`, then commit the results, logs, collections, checkpoints and new caches.\n")
    (DOCS / "PAPER_SONDEO.md").write_text(
        "# Selector paper: the sondeo, fold 0 of every dataset with every oracle\n\n"
        f"{header} Datasets {', '.join(DATASETS)} with the first paper's oracle, the margin oracle (τ = {MARGIN}) and the "
        f"robust one (edge perturbations {ROBUST}); and tri ({', '.join(TRI)}) with its exact rule and the three oracles on a GCN. "
        "colors-3 on 20 fixed fold-0 instances. bbbp's original-oracle fold 0 runs locally. "
        "What already has results in git is skipped.\n\n" + steps)
    (DOCS / "PAPER_SONDEO_A.md").write_text("# Sondeo, queue A: collections, bls_v0 and bls_v1 on fold 0 (see PAPER_SONDEO.md)\n\n"
                                            + "\n".join(f"- [ ] {c}" for c in sondeo_a) + "\n")
    (DOCS / "PAPER_SONDEO_B.md").write_text("# Sondeo, queue B: selector_v1 on fold 0, after the pretraining (see PAPER_SONDEO.md)\n\n"
                                            + "\n".join(f"- [ ] {c}" for c in sondeo_b) + "\n")
    (DOCS / "PAPER_LOCAL_3.md").write_text(
        f"# Selector paper, local queue 3: fold 0 again after the generator fixes: {', '.join(LOCAL_TCR)} (collections, bls_v0, bls_v1, control) and {', '.join(LOCAL_REDO)} (collections, bls_v1)\n\n"
        f"{header} Run after `PAPER_LOCAL_2.md`: `python scripts/run_revision_queue.py --list docs/selector/PAPER_LOCAL_3.md --workers 1 --timeout 86400`, "
        f"then `python scripts/pretrain_selector_checkpoints.py --datasets {' '.join(LOCAL_TCR + LOCAL_REDO)}`, then `PAPER_LOCAL_4.md`.\n\n" + "\n".join(f"- [ ] {c}" for c in local_tcr) + "\n")
    (DOCS / "PAPER_LOCAL_4.md").write_text(
        f"# Selector paper, local queue 4: selector_v1 on fold 0 of {', '.join(LOCAL_REDO)} (needs the checkpoints of PAPER_LOCAL_3's collections)\n\n"
        f"{header}\n\n" + "\n".join(f"- [ ] {c}" for c in local_redo) + "\n")
    for old in DOCS.glob("PAPER_Q*.md"):
        old.unlink()
    index, n = [], 0
    for phase, cfgs, needs in (("S1", s1, ""), ("S2", s2, "committed checkpoints"), ("S3", s3, "committed caches of " + ", ".join(PENDING_CACHES) + " and their checkpoints")):
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
        "(`launchers/slurm_submit.sh -p normal -m 10 <queue>`). Every dataset on every fold; what the local preliminary "
        "(`PAPER_LOCAL_1.md`, `PAPER_LOCAL_2.md`) already produced is skipped once committed. S1 needs nothing; S2 needs the checkpoints of "
        f"`scripts/pretrain_selector_checkpoints.py` committed and synced; S3 needs the committed dataset and oracle caches of {', '.join(PENDING_CACHES)} "
        "(bzr comes out of the local preliminary, enzymes from the server's warm job, `docs/rerun/README.md`, section 2) "
        "and their checkpoints.\n\n| Queue | Phase | Configs | Needs | Cells (dataset/generator/arm) |\n|---|---|---|---|---|\n"
        + "\n".join(index) + "\n")
    print(f"sondeo: A {len(sondeo_a)} + B {len(sondeo_b)}; local 1: {len(local1)}, local 2: {len(local2)}, local 3: {len(local_tcr)}, server: S1 {len(s1)} + S2 {len(s2)} + S3 {len(s3)} in {n} queues")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
