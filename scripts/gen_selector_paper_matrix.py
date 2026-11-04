#!/usr/bin/env python3
"""The selector paper's experiment matrix: seven datasets with two oracles each plus the tri family,
ten folds, DCE, the arms below, and the collections that pretrain the selector without leakage.

Final shape (user's decisions of 2026-11-03, after ``docs/selector/ANALISIS_SONDEO.md``): the paper is
bls_v0, bls_v1, selector_v1 and the untrained control, no selector_v2; the robust oracle is out of the
experiments (its code, trees and sondeo results stay); the margin oracle uses τ = 0.7. The server runs the
fleet in batches sized by estimated hours (``HOURS``, from the sondeo), 10 jobs at a time, about two
days each (``PAPER_FLOTA.md`` is the tutorial, ``PAPER_B<nn>.md`` the batches).

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

Cells (``cells()``): every dataset of ``DATASETS`` with the first paper's oracle (``<dataset>``) and with a
confidence margin around it (``<dataset>-m70``, MarginOracle); colors-3 always as its 20-instance
pseudo-dataset (``colors-3<variant>-s20``, the evaluator's ``instance_ids``, hash-neutral, 40 instances on
the collection folds); and tri (``TRI``), each configuration with its exact rule, a GCN, and the GCN with
margin (``TRI_EXPS``).

Pretraining without leakage: the checkpoint used on fold f must not have seen f's test instances, so each
cell has two, trained on the collections of folds 1 and 2: ``<CKPT_PREFIX>_<cell>-f1.pt`` serves every fold
but fold 1, ``-f2.pt`` serves fold 1 (``scripts/pretrain_selector_checkpoints.py`` builds whichever is missing).

Queues (``docs/selector/PAPER_*.md``), in the order they can run (server, at most 10 jobs at a time):

    PAPER_FLOTA.md     the tutorial: one batch at a time, the pretraining job after the first batch
    PAPER_B<nn>.md     the batches, each about ``BATCH_HOURS`` job-hours (two days at 10 jobs): the
                       collections first, then bls_v0, bls_v1 and the control cell by cell, then
                       selector_v1 (which needs the committed checkpoints). What already has results in
                       git is skipped by ``slurm_submit.sh``
    PAPER_QUEUES.md    the index, with the estimated hours

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
SUBSETS = {"colors-3-s20": "colors-3", "colors-3-m70-s20": "colors-3-m70"}   # pseudo-dataset -> dataset tree
SUBSET_EVAL, SUBSET_COLLECT = 20, 40      # sampled test instances per fold: evaluation arms, collection arms
# The two oracles of the paper's tables: the first paper's (the thesis trees) and the same with a confidence
# margin (MarginOracle, <dataset>-m70; τ = 0.9 made PROTEINS unsolvable, ANALISIS_SONDEO.md section 7).
# The robust oracle (OracleTorch edge_perturbation, the <dataset>-rob trees) is out of the experiments since
# 2026-11-03: it changed little except in bbbp. Its trees and sondeo results stay; set VARIANTS["rob"] to
# bring it back.
MARGIN = 0.7
ROBUST = {"drop": 0.1, "add": 0.1}
VARIANTS = {"m70": list(DATASETS)}
# tri, the triangle-threshold synthetic, has its own table: several configurations, each with the exact rule
# as reference and the oracles on a GCN (with each node's triangle count as a feature): <tri>-gcn and
# <tri>-gcn-m70. All of one configuration share the dataset cache and folds.
TRI = {"tri-40": "TRI-500-40", "tri-40n": "TRI-500-40n", "tri-80": "TRI-500-80"}
TRI_EXPS = [e for t in TRI for e in (t, f"{t}-gcn", *(f"{t}-gcn-{v}" for v in VARIANTS))]
CACHE_PREFIX = {"colors-3": "COLORS-3", "colors-3-m70": "COLORS-3"}   # dataset cache file prefix of each subset's tree
N_FOLDS = {}       # datasets with fewer than 10 folds; none
DATASET_FLAGS = ["attributed", "recompute_features", "neigh_factor", "runtime_factor", "max_runtime", "max_neigh"]
SEVEN = ["identity", "average_smoothing", "weighted_smoothing", "laplacian_regularization", "feature_aggregation",
         "heat_kernel_diffusion", "random_walk_diffusion"]
BUDGET, COLLECT_BUDGET, COLLECT_FOLDS = 10000, 2000, (1, 2)
CACHE = "lab/data/cache/generator_outputs"
BATCH_HOURS = 420          # job-hours per batch: two days at 10 jobs with some slack for the longest jobs
JOB_OVERHEAD_H = 0.3       # start-up of every job (environment, dataset and oracle load, generator cache)
SKELETON_FACTOR = 1.5      # generation 2 lets the blind fallback spend the whole budget on stuck instances
# Hours per config measured in the sondeo (fold 0, 2026-10-31; the Runtime metric summed over the
# instances): (bls_v0, bls_v1, selector_v1, collection). The m70 cells take the larger of the original and
# the m90 cell; bbbp's v0 the m90 one; colors-3 the DCE scan of its 20 instances, paid by the first job of
# each fold. The untrained control counts as bls_v1.
HOURS = {"proteins": (3.1, 4.2, 5.5, 0.95), "proteins-m70": (4.6, 4.2, 6.3, 1.2),
         "synthie": (0.09, 0.06, 0.12, 0.04), "synthie-m70": (0.16, 0.11, 0.30, 0.09),
         "bbbp": (1.5, 0.33, 0.41, 0.40), "bbbp-m70": (1.5, 1.35, 3.22, 0.86),
         "aids": (1.5, 0.89, 0.77, 0.46), "aids-m70": (1.5, 1.17, 3.17, 0.85),
         "bzr": (0.07, 0.03, 0.03, 0.02), "bzr-m70": (0.08, 0.06, 0.07, 0.04),
         "enzymes": (0.13, 0.09, 0.23, 0.08), "enzymes-m70": (0.14, 0.11, 0.15, 0.09),
         "colors-3-s20": (0.9, 0.9, 0.9, 0.9), "colors-3-m70-s20": (0.9, 0.9, 0.9, 0.9),
         "tri-40": (0.01, 0.01, 0.03, 0.06), "tri-40-gcn": (0.03, 0.02, 0.06, 0.06), "tri-40-gcn-m70": (0.04, 0.03, 0.04, 0.09),
         "tri-40n": (0.01, 0.01, 0.02, 0.04), "tri-40n-gcn": (0.02, 0.01, 0.02, 0.05), "tri-40n-gcn-m70": (0.03, 0.02, 0.05, 0.07),
         "tri-80": (0.02, 0.03, 0.13, 0.13), "tri-80-gcn": (0.10, 0.08, 0.30, 0.15), "tri-80-gcn-m70": (0.10, 0.09, 0.14, 0.16)}
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
        for v in ("", *(f"-{v}" for v in VARIANTS)):
            out.append(f"colors-3{v}-s20" if ds == "colors-3" else f"{ds}{v}")
    return out + list(TRI_EXPS)


def est_hours(ds: str, arm: str) -> float:
    v0, v1, sel, coll = HOURS[ds]
    base = {V0: v0, V1: v1 * SKELETON_FACTOR, UNT1: v1 * SKELETON_FACTOR, SEL1: sel * SKELETON_FACTOR, COLL: coll}[arm]
    return base + JOB_OVERHEAD_H


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
    all_cells = cells()
    first, rest, last = [], [], []        # (config, hours): collections; v0, v1, control; selector_v1
    for ds in all_cells:
        first += [(write(skeleton_config(ds, COLL, f), ds, COLL, f), est_hours(ds, COLL)) for f in COLLECT_FOLDS]
        for arm in (V0, V1, UNT1):
            rest += [(write(v0_config(ds, f) if arm == V0 else skeleton_config(ds, arm, f), ds, arm, f), est_hours(ds, arm)) for f in folds(ds)]
        last += [(write(skeleton_config(ds, SEL1, f), ds, SEL1, f), est_hours(ds, SEL1)) for f in folds(ds)]
    done = {c for c, _ in first + rest + last if _results_exist(c)}
    batches, cur, cur_h = [], [], 0.0
    for c, h in first + rest + last:
        if c in done:
            continue
        if cur and cur_h + h > BATCH_HOURS:
            batches.append((cur, cur_h)); cur, cur_h = [], 0.0
        cur.append(c); cur_h += h
    if cur:
        batches.append((cur, cur_h))
    sel_batch = next((i for i, (b, _) in enumerate(batches, 1) if any(f"/{SEL1}/" in c for c in b)), None)
    for old in list(DOCS.glob("PAPER_Q*.md")) + list(DOCS.glob("PAPER_B*.md")) + list(DOCS.glob("PAPER_SONDEO*.md")):
        old.unlink()
    header = "Generated by `scripts/gen_selector_paper_matrix.py` (its docstring gives the arms and the order)."
    index = []
    for i, (b, h) in enumerate(batches, 1):
        name = f"PAPER_B{i:02d}.md"
        arms = sorted({c.split("/")[5] for c in b})
        needs = "the committed checkpoints" if any(f"/{SEL1}/" in c for c in b) else ""
        (DOCS / name).write_text(f"# Selector paper, batch {i:02d}: about {h:.0f} job-hours, {h / 10 / 24:.1f} days at 10 jobs{', needs ' + needs if needs else ''}\n\n"
                                 f"{header} `launchers/slurm_submit.sh -p normal -m 10 docs/selector/{name}`.\n\n"
                                 + "\n".join(f"- [ ] {c}" for c in b) + "\n")
        cs = sorted({c.split("/")[3] for c in b})
        index.append(f"| `{name}` | {len(b)} | {h:.0f} | {h / 10 / 24:.1f} | {needs} | {', '.join(arms)} | {', '.join(cs)} |")
    total = sum(h for _, h in batches)
    (DOCS / "PAPER_QUEUES.md").write_text(
        "# Selector paper: the server batches\n\nGenerated by `scripts/gen_selector_paper_matrix.py`; the tutorial is `PAPER_FLOTA.md`. "
        f"{len(all_cells)} cells, every fold, {sum(len(b) for b, _ in batches)} configs still to run ({len(done)} already have results), "
        f"about {total:.0f} job-hours, {total / 10 / 24:.1f} days at 10 jobs. Hours are the sondeo's per-config times (`HOURS`), "
        f"times {SKELETON_FACTOR} for the skeleton arms, plus {JOB_OVERHEAD_H} h per job.\n\n"
        "| Batch | Configs | Job-hours | Days at 10 | Needs | Arms | Cells |\n|---|---|---|---|---|---|---|\n" + "\n".join(index) + "\n")
    steps = [
        "Rule: on the server only Slurm jobs compute (nothing heavy on the login node), and at most 10 jobs at a time. "
        "Every batch is one job array throttled to 10 tasks: Slurm starts the next task as soon as one ends, so a batch "
        "needs no attention until `squeue -u $USER` is empty. A batch that ends with failed tasks is submitted again as it is: "
        "finished configs are skipped. No cache job is needed: the margin oracles wrap the committed base oracles.\n",
        "**Step 0. Sync** (only git and the pinned environment):\n\n```bash\ncd ~/projects/GRETEL\nlaunchers/server_sync.sh\nconda activate GRTL\n```\n",
        f"**Step 1. Batch 1** ({len(batches[0][0])} configs, about {batches[0][1] / 10 / 24:.1f} days): the collections of folds 1 and 2 of every cell, "
        "then bls_v0, bls_v1 and the untrained control cell by cell:\n\n```bash\nlaunchers/slurm_submit.sh -p normal -m 10 docs/selector/PAPER_B01.md\n"
        "squeue -u $USER                                                        # empty: the batch is over\n"
        "grep -lE \"exit status [1-9]|Traceback\" output/logs/*/*/*_<job>_*.out     # <job>: the array number slurm_submit printed; failed tasks, if any\n```\n",
        "**Step 2. Pretraining** (1 job, both collection folds of every cell, about an hour), as soon as batch 1 is over:\n\n"
        "```bash\nsbatch -p normal launchers/slurm_pretrain.sh " + " ".join(all_cells) + "\nsqueue -u $USER\n"
        "tail -1 output/slurm/pretrain_<job>.out     # built 46 checkpoint(s); collections still missing: none\n```\n\n"
        "Then on the local machine, so that every clone has the checkpoints and the results so far:\n\n"
        "```bash\nmake server-import\ngit add lab/output/results lab/output/queue_logs lab/output/selector_logs models/pretrained && git commit -m \"Selector paper: batch 1 and the checkpoints\" && git push\n```\n\n"
        "and on the server `launchers/server_sync.sh` again.\n",
    ]
    for i, (b, h) in enumerate(batches[1:], 2):
        steps.append(f"**Step {i + 1}. Batch {i}** ({len(b)} configs, about {h / 10 / 24:.1f} days"
                     + (", selector_v1 starts here: it needs the checkpoints of step 2 synced" if i == sel_batch else "") + "):\n\n"
                     f"```bash\nlaunchers/slurm_submit.sh -p normal -m 10 docs/selector/PAPER_B{i:02d}.md\nsqueue -u $USER\n```\n")
    steps.append("**Last step. Back home** after every batch, or at the end: `make server-import`, commit the results and logs "
                 "(`git add lab/output/results lab/output/queue_logs lab/output/selector_logs`), and the tables: "
                 "`python scripts/sondeo_summary.py --gen 2 --fold <f>` gives one row per cell for one fold, "
                 "`scripts/analyze_selector_arms.py --arms v0=<cell>-sel_dce_lcls-v0 v1=<cell>-sel_dce_lcls-v1 sel=<cell>-sel_dce_lcls-sel-v1 "
                 "untrained=<cell>-sel_dce_lcls-sel-v1-untrained --monitor sel=sel-v1-<cell> untrained=sel-v1-untrained-<cell> v1=v1-<cell> "
                 "--dataset <dataset cache name> --folds 0 1 2 3 4 5 6 7 8 9` the full study of one cell.\n")
    (DOCS / "PAPER_FLOTA.md").write_text(
        "# Selector paper: running every experiment on the server\n\n"
        f"{header} The paper's arms are bls_v0, bls_v1, selector_v1 and the untrained control on {len(all_cells)} cells "
        f"({', '.join(DATASETS)} with the first paper's oracle and with the margin oracle τ = {MARGIN}, colors-3 on {SUBSET_EVAL} fixed "
        f"instances per fold, and tri: {', '.join(TRI)} with the exact rule, a GCN and the GCN with margin), ten folds, "
        f"10 000 calls. {sum(len(b) for b, _ in batches)} configs in {len(batches)} batches (`PAPER_QUEUES.md`), one batch at a time.\n\n"
        + "\n".join(steps))
    print(f"{len(all_cells)} cells; {len(done)} configs with results; {len(batches)} batches: "
          + ", ".join(f"B{i:02d} {len(b)} cfg {h:.0f} h" for i, (b, h) in enumerate(batches, 1)) + f"; selector_v1 from batch {sel_batch}")
    return 0


def _results_exist(rel_cfg: str) -> bool:
    conf = json.load(open(REPO / rel_cfg))
    scope, fold = conf["experiment"]["scope"], re.search(r"generate_minimize(\d+)\.jsonc$", rel_cfg).group(1)
    root = REPO / "lab/output/results" / scope
    return root.is_dir() and any(root.rglob(f"results_{fold}_1.json"))


if __name__ == "__main__":
    raise SystemExit(main())
