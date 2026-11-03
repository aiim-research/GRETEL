#!/usr/bin/env python3
"""The selector paper's experiment matrix: seven datasets with three oracles each plus the tri family,
ten folds, DCE, the arms below, and the collections that pretrain the selector without leakage.

Generation 2 of the arms (2026-11-03, after the sondeo, ``docs/selector/ANALISIS_SONDEO.md``): the block
ladder is off and the blind fallback may spend the whole remaining budget (``KNOBS``), every monitor CSV
is named after its cell (``model_tag``), the collections also store edge identifiers (``collect_ids``),
and the margin oracle uses τ = 0.7. The sondeo's arms (``dce-lcls-sel-random``, ``dce-lcls-sel``,
``dce-lcls-sel-untrained``, ``dce-lcls-sel-collect``, the ``<dataset>-m90`` trees) and their results stay
as they are, as the record of that study, and are not regenerated here.

Arms (names in the paper in brackets), each in ``lab/config/generate_minimize/<cell>-sel/dce/``:

    dce-lcls-v0                 [bls_v0]        the thesis implementation as the paper describes it: removal
                                                levels from one edge (code fix), stop after the first outer
                                                iteration without improvement, the seven attribute strategies
    dce-lcls-v1                 [bls_v1]        the selector skeleton with uniform proposals and no model
    dce-lcls-sel-v1             [selector_v1]   the pretrained, frozen selector
    dce-lcls-sel-v1-untrained   [control]       the same network and policy, untrained
    dce-lcls-collect2                           uniform proposals dumping every scored example with its edge
                                                identifiers (folds 1 and 2 only, budget 2000), for the
                                                pretraining and the offline design of selector_v2

The skeleton arms take the minimizer settings of the PROTEINS study (the templates in ``proteins-sel/dce``,
``ARMS``), the cell's thesis config for everything else, and the flags that config sets about how the
oracle is queried (``DATASET_FLAGS``).

Cells (``cells()``): every dataset of ``DATASETS`` with the first paper's oracle (``<dataset>``), with a
confidence margin around it (``<dataset>-m70``, MarginOracle) and retrained with random edge perturbations
(``<dataset>-rob``); colors-3 always as its 20-instance pseudo-dataset (``colors-3<variant>-s20``, the
evaluator's ``instance_ids``, hash-neutral, 40 instances on the collection folds); and tri (``TRI``), each
configuration with its exact rule, a GCN, and the GCN with margin and robust (``TRI_EXPS``).

Pretraining without leakage: the checkpoint used on fold f must not have seen f's test instances, so each
cell has two, trained on the collections of folds 1 and 2: ``<CKPT_PREFIX>_<cell>-f1.pt`` serves every fold
but fold 1, ``-f2.pt`` serves fold 1 (``scripts/pretrain_selector_checkpoints.py`` builds whichever is missing).

Queues (``docs/selector/PAPER_*.md``), in the order they can run (server, at most 10 jobs at a time):

    PAPER_SONDEO.md    the tutorial of the sondeo: fold 0 of every cell. PAPER_SONDEO_A.md (collections of
                       folds 1 and 2, bls_v0 and bls_v1 on fold 0), the pretraining job, PAPER_SONDEO_B.md
                       (selector_v1 on fold 0). What already has results in git is skipped
    PAPER_Q<nn>.md     the fleet, at most 30 configs each, every cell on every fold: S1 needs nothing
                       (collections, v0, bls_v1, control), S2 needs the committed checkpoints (selector_v1)
    PAPER_QUEUES.md    the index

    python scripts/gen_selector_paper_matrix.py
"""
from __future__ import annotations

import json
import pickle
import re
from collections import OrderedDict, namedtuple
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TREE = REPO / "lab/config/generate_minimize"
PROTEINS = TREE / "proteins-sel/dce"
DOCS = REPO / "docs/selector"
# The datasets of the first paper. Tree-Cycles and ASD are stale since 2026-10-30 (their configs stay).
# synthie, colors-3, enzymes and bzr are "no harm" rows (user's decision 2026-11-03): every arm reaches a
# 1 to 3 edge counterfactual with every oracle, they stay in the tables to show the selector costs nothing.
DATASETS = ["proteins", "synthie", "bbbp", "aids", "colors-3", "bzr", "enzymes"]
SUBSETS = {"colors-3-s20": "colors-3", "colors-3-m70-s20": "colors-3-m70", "colors-3-rob-s20": "colors-3-rob"}   # pseudo-dataset -> dataset tree
SUBSET_EVAL, SUBSET_COLLECT = 20, 40      # sampled test instances per fold: evaluation arms, collection arms
# The three oracles of the paper's tables: the first paper's (the thesis trees), the same with a confidence
# margin (MarginOracle, <dataset>-m70; τ = 0.9 made PROTEINS unsolvable, ANALISIS_SONDEO.md section 7) and
# one retrained with random edge perturbations (OracleTorch edge_perturbation, <dataset>-rob).
MARGIN = 0.7
ROBUST = {"drop": 0.1, "add": 0.1}
VARIANTS = {"m70": list(DATASETS), "rob": list(DATASETS)}
# tri, the triangle-threshold synthetic, has its own table: several configurations, each with the exact rule
# as reference and the three oracles on a GCN (with each node's triangle count as a feature):
# <tri>-gcn, <tri>-gcn-m70, <tri>-gcn-rob. All of one configuration share the dataset cache and folds.
TRI = {"tri-40": "TRI-500-40", "tri-40n": "TRI-500-40n", "tri-80": "TRI-500-80"}
TRI_EXPS = [e for t in TRI for e in (t, f"{t}-gcn", f"{t}-gcn-m70", f"{t}-gcn-rob")]
CACHE_PREFIX = {"colors-3": "COLORS-3", "colors-3-m70": "COLORS-3", "colors-3-rob": "COLORS-3"}   # dataset cache file prefix of each subset's tree
N_FOLDS = {}       # datasets with fewer than 10 folds; none
DATASET_FLAGS = ["attributed", "recompute_features", "neigh_factor", "runtime_factor", "max_runtime", "max_neigh"]
SEVEN = ["identity", "average_smoothing", "weighted_smoothing", "laplacian_regularization", "feature_aggregation",
         "heat_kernel_diffusion", "random_walk_diffusion"]
BUDGET, COLLECT_BUDGET, COLLECT_FOLDS = 10000, 2000, (1, 2)
CACHE = "lab/data/cache/generator_outputs"
QUEUE_SIZE = 30
# Generation 2 of the skeleton arms: name -> (template arm in proteins-sel/dce, monitor tag)
Arm = namedtuple("Arm", "template tag")
V0, V1, SEL1, UNT1, COLL = "dce-lcls-v0", "dce-lcls-v1", "dce-lcls-sel-v1", "dce-lcls-sel-v1-untrained", "dce-lcls-collect2"
ARMS = {V1: Arm("dce-lcls-sel-random", "v1"), SEL1: Arm("dce-lcls-sel", "sel-v1"),
        UNT1: Arm("dce-lcls-sel-untrained", "sel-v1-untrained"), COLL: Arm("dce-lcls-sel-random", "collect2")}
# ANALISIS_SONDEO.md section 5: no block ladder (a run-scoped switch that traded GED for calls in some arms
# and not in others of the same cell), and the blind fallback of an instance where nothing worked may use
# the whole remaining budget, as bls_v0 does.
KNOBS = {"block_removal": False, "overshoot_budget_frac": 1.0}
COLLECT_NAME = "collect2"                  # lab/output/selector_pretrain/<cell>_collect2-f<f>.pt
CKPT_PREFIX = "edge_selector_sel2-pre"     # models/pretrained/edge_selector_sel2-pre_<cell>-f<f>.pt


def cells() -> list[str]:
    out = []
    for ds in DATASETS:
        for v in ("", "-m70", "-rob"):
            out.append(f"colors-3{v}-s20" if ds == "colors-3" else f"{ds}{v}")
    return out + list(TRI_EXPS)


def base(ds: str) -> str:
    return SUBSETS.get(ds, ds)


def folds(ds: str) -> range:
    return range(N_FOLDS.get(base(ds), 10))


def ckpt(ds: str, fold: int) -> str:
    return f"models/pretrained/{CKPT_PREFIX}_{ds}-f{2 if fold == 1 else 1}.pt"


def template_dir(ds: str) -> Path:
    return PROTEINS / "dce-lcls" if ds == "proteins" else TREE / ds / "dce/dce-lcls"


def thesis_config(ds: str, fold: int) -> dict:
    return json.load(open(template_dir(base(ds)) / f"generate_minimize{fold}.jsonc"))


def subset_ids(ds: str, fold: int, n: int) -> list[int]:
    """A fixed random sample of the fold's test instances, from the committed dataset cache (TU datasets
    number their instances by position, so split indices are instance ids). The same seed per fold, so
    the collection's 40 and the evaluation's 20 are drawn from the same stream."""
    name = SUBSETS[ds]
    caches = list((REPO / "lab/data/cache/datasets").glob(f"{CACHE_PREFIX[name]}*"))
    assert len(caches) == 1, f"{name}: expected one committed dataset cache, found {caches}"
    splits = pickle.load(open(caches[0], "rb"))["splits"]
    return sorted(random.Random(1000 + fold).sample(sorted(splits[fold]["test"]), n))


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
    if ds in SUBSETS:
        conf["evaluator"]["parameters"]["instance_ids"] = subset_ids(ds, fold, SUBSET_EVAL)
    expl = conf["doe-triplets"][0]["explainer"]["parameters"]
    expl["generator_cache"] = CACHE
    assert expl["minimizer"]["class"].endswith("local_search.local_search.LocalSearch"), expl["minimizer"]["class"]
    par = expl["minimizer"]["parameters"]
    par.update({"seed": 0, "max_oracle_calls": BUDGET, "strict_budget": True,
                "stop_after_failed_iteration": True, "attribute_methods": list(SEVEN)})
    return conf


def skeleton_config(ds: str, arm: str, fold: int) -> dict:
    """A selector-skeleton arm on the cell's thesis config, PROTEINS minimizer settings, generation-2 knobs."""
    src = json.load(open(PROTEINS / ARMS[arm].template / "generate_minimize0.jsonc"))
    conf = thesis_config(ds, fold)
    set_fold(conf, fold)
    conf["experiment"]["scope"] = f"{ds}-sel_" + arm.replace("dce-lcls", "dce_lcls", 1)
    conf["evaluator"]["parameters"].pop("instance_ids", None)
    if ds in SUBSETS:
        conf["evaluator"]["parameters"]["instance_ids"] = subset_ids(ds, fold, SUBSET_COLLECT if arm == COLL else SUBSET_EVAL)
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
    par.update(KNOBS)
    par["model_tag"] = f"{ARMS[arm].tag}-{ds}"
    par["max_oracle_calls"] = COLLECT_BUDGET if arm == COLL else BUDGET
    par.pop("init_checkpoint", None)
    if arm == SEL1:
        par["init_checkpoint"] = ckpt(ds, fold)
    if arm == COLL:
        par["collect_examples"] = f"lab/output/selector_pretrain/{ds}_{COLLECT_NAME}-f{fold}.pt"
        par["collect_ids"] = True
    return conf


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
    make_tri_templates()
    for v, dss in VARIANTS.items():
        for ds in dss:
            make_variant(ds, v)
    for t in TRI:
        for v in VARIANTS:
            make_variant(f"{t}-gcn", v)
    sondeo_a, sondeo_b, s1, s2 = [], [], [], []
    all_cells = cells()
    for ds in all_cells:
        v0 = {f: write(v0_config(ds, f), ds, V0, f) for f in folds(ds)}
        coll = {f: write(skeleton_config(ds, COLL, f), ds, COLL, f) for f in COLLECT_FOLDS}
        v1 = {f: write(skeleton_config(ds, V1, f), ds, V1, f) for f in folds(ds)}
        unt = {f: write(skeleton_config(ds, UNT1, f), ds, UNT1, f) for f in folds(ds)}
        sel = {f: write(skeleton_config(ds, SEL1, f), ds, SEL1, f) for f in folds(ds)}
        sondeo_a += [coll[1], coll[2], v0[0], v1[0]]
        sondeo_b.append(sel[0])
        s1 += list(coll.values()) + list(v0.values()) + list(v1.values()) + list(unt.values())
        s2 += list(sel.values())

    header = "Generated by `scripts/gen_selector_paper_matrix.py` (its docstring gives the arms and the order)."
    steps = (
        "Rule: on the server only Slurm jobs compute (nothing heavy on the login node), and at most 10 jobs at a time. "
        "Each step starts once `squeue -u $USER` shows nothing left of the previous one. No cache job is needed: the "
        "margin oracles wrap the committed base oracles, and the robust ones are committed.\n\n"
        "**Step 0. Sync** (only git and the pinned environment):\n\n```bash\ncd ~/projects/GRETEL\nlaunchers/server_sync.sh\n"
        "conda activate GRTL\n```\n\n"
        f"**Step 1. Queue A**: the collections of folds 1 and 2, bls_v0 and bls_v1 on fold 0 ({len(sondeo_a)} configs, 10 at a time; "
        "the bls_v0 configs that already have results are skipped):\n\n"
        "```bash\nlaunchers/slurm_submit.sh -p normal -m 10 docs/selector/PAPER_SONDEO_A.md\nsqueue -u $USER\n"
        "grep -lE \"exit status [1-9]|Traceback\" output/logs/*/*/*_<job>_*.out     # <job>: the array job number slurm_submit printed\n```\n\n"
        "A queue that ends with failures can be submitted again as it is: finished configs are skipped.\n\n"
        "**Step 2. Pretraining** of the selector checkpoints from those collections (1 job, both folds of every cell):\n\n"
        "```bash\nsbatch -p normal launchers/slurm_pretrain.sh " + " ".join(all_cells) + "\nsqueue -u $USER\ntail output/slurm/pretrain_<job>.out     # ends with: built N checkpoint(s); collections still missing: none\n```\n\n"
        f"**Step 3. Queue B**: selector_v1 on fold 0 ({len(sondeo_b)} configs):\n\n"
        "```bash\nlaunchers/slurm_submit.sh -p normal -m 10 docs/selector/PAPER_SONDEO_B.md\nsqueue -u $USER\n```\n\n"
        "**Step 4. Back home** (local machine): `make server-import`, then commit the results, logs, checkpoints and new caches, "
        "and `python scripts/sondeo_summary.py` for the table.\n")
    (DOCS / "PAPER_SONDEO.md").write_text(
        "# Selector paper: the sondeo (generation 2 of the arms), fold 0 of every cell\n\n"
        f"{header} Datasets {', '.join(DATASETS)} with the first paper's oracle, the margin oracle (τ = {MARGIN}) and the "
        f"robust one (edge perturbations {ROBUST}); and tri ({', '.join(TRI)}) with its exact rule and the three oracles on a GCN. "
        f"colors-3 on {SUBSET_EVAL} fixed instances per fold ({SUBSET_COLLECT} on the collection folds). "
        "The arms carry the configuration decided after the first sondeo (`ANALISIS_SONDEO.md`, section 5) and new names, so "
        "everything but bls_v0 runs again; the first sondeo's results stay under their old scopes. "
        "What already has results in git is skipped.\n\n" + steps)
    (DOCS / "PAPER_SONDEO_A.md").write_text("# Sondeo, queue A: collections (folds 1 and 2), bls_v0 and bls_v1 on fold 0 (see PAPER_SONDEO.md)\n\n"
                                            + "\n".join(f"- [ ] {c}" for c in sondeo_a) + "\n")
    (DOCS / "PAPER_SONDEO_B.md").write_text("# Sondeo, queue B: selector_v1 on fold 0, after the pretraining (see PAPER_SONDEO.md)\n\n"
                                            + "\n".join(f"- [ ] {c}" for c in sondeo_b) + "\n")
    for old in DOCS.glob("PAPER_Q*.md"):
        old.unlink()
    index, n = [], 0
    for phase, cfgs, needs in (("S1", s1, ""), ("S2", s2, "committed checkpoints")):
        for i in range(0, len(cfgs), QUEUE_SIZE):
            n += 1
            chunk = cfgs[i:i + QUEUE_SIZE]
            name = f"PAPER_Q{n:02d}.md"
            (DOCS / name).write_text(f"# Selector paper, server queue {n:02d} (phase {phase}{', needs ' + needs if needs else ''})\n\n"
                                     f"{header} `launchers/slurm_submit.sh -p normal -m 10 docs/selector/{name}`.\n\n"
                                     + "\n".join(f"- [ ] {c}" for c in chunk) + "\n")
            cs = sorted({"/".join(c.split("/")[3:6]) for c in chunk})
            index.append(f"| `{name}` | {phase} | {len(chunk)} | {needs} | {', '.join(f'`{c}`' for c in cs)} |")
    (DOCS / "PAPER_QUEUES.md").write_text(
        "# Selector paper: server queues (the fleet)\n\nGenerated by `scripts/gen_selector_paper_matrix.py`, in submission order, one at a time "
        "(`launchers/slurm_submit.sh -p normal -m 10 <queue>`). Every cell on every fold; what the sondeo (`PAPER_SONDEO.md`) "
        "already produced is skipped once committed. S1 needs nothing; S2 needs the checkpoints of both collection folds "
        "(`scripts/pretrain_selector_checkpoints.py`) committed and synced. The selector_v2 arm is added here once it exists "
        "(`ANALISIS_SONDEO.md`, section 6).\n\n| Queue | Phase | Configs | Needs | Cells (dataset/generator/arm) |\n|---|---|---|---|---|\n"
        + "\n".join(index) + "\n")
    print(f"{len(all_cells)} cells; sondeo: A {len(sondeo_a)} + B {len(sondeo_b)}; fleet: S1 {len(s1)} + S2 {len(s2)} in {n} queues")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
