#!/usr/bin/env python3
"""Generate config files for the REVISION_EXPERIMENTS.md article matrix.

Decoupled protocol (see REVISION_EXPERIMENTS.md):
  * Each (dataset, generator, minimizer) runs ONCE, no seed in the scope
    name, but with a FIXED internal seed = 0 (reproducible; this no-seed run
    *is* "seed 0"). Scope: ``<ds>_<gen>_<min>``.
  * Multiple seeds ONLY for LBS+DCE (E1b stability): besides the no-seed
    (=seed 0) run, ``dce-lcls-seed{1,2,3}`` with internal seed 1/2/3.
    Scope: ``<ds>_dce_lcls_seed<N>``.

Seed parameter key per component (matches each class' init):
  * generator: ofs/rsgg -> ``seed``; dfs -> ``random_seed``; dce -> none.
  * minimizer: lcls/obs/rhc -> ``seed``; dbs -> ``random_seed``.

``recompute_features: false`` is set on feature-blind datasets (asd,
tcr-tco-300) whose oracle ignores recomputed node features.

Run from the repo root::

    python scripts/gen_revision_configs.py            # regenerate everything
    python scripts/gen_revision_configs.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CFG_ROOT = REPO / "lab" / "config" / "generate_minimize"

DATASETS = ["tcr-tco-300", "asd", "synthie", "bbbp"]
GENERATORS = ["dce", "ofs", "dfs", "rsgg"]
MINIMIZERS = ["lcls", "obs", "dbs", "rhc"]
FEATURE_BLIND = {"asd", "tcr-tco-300"}
LBS_SEEDS = [1, 2, 3]            # extra seeds, dce-lcls only (no-seed = seed 0)

GEN_SEED_KEY = {"dce": None, "ofs": "seed", "rsgg": "seed", "dfs": "random_seed"}
MIN_SEED_KEY = {"lcls": "seed", "obs": "seed", "dbs": "random_seed", "rhc": "seed"}

RHC_CLASS = "src.explainer.future.metaheuristic.local_search.random_hill_climbing.RandomHillClimbing"
SEEDED_DIR_RE = re.compile(r"^(dce|ofs|dfs|rsgg)-(lcls|obs|dbs|rhc)-seed\d+$")


def _read_jsonc(path: Path):
    txt = path.read_text()
    txt = re.sub(r"/\*.*?\*/", "", txt, flags=re.DOTALL)
    txt = re.sub(r"//.*?\n", "\n", txt)
    return json.loads(txt, object_pairs_hook=OrderedDict)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)


def _src_dir(ds, gen, mn):
    """Template dir for a combo; rhc clones from <gen>-lcls."""
    d = CFG_ROOT / ds / gen / f"{gen}-{mn}"
    if d.is_dir():
        return d
    if mn == "rhc":
        return CFG_ROOT / ds / gen / f"{gen}-lcls"
    return None


def _apply(cfg, ds, gen, mn, scope, seed):
    """Patch a cloned config: scope, generator+minimizer seed, rhc swap,
    recompute_features for feature-blind datasets."""
    cfg["experiment"]["scope"] = scope
    for t in cfg["doe-triplets"]:
        p = t["explainer"]["parameters"]
        gk = GEN_SEED_KEY[gen]
        if gk:
            p["generator"]["parameters"][gk] = int(seed)
        minim = p["minimizer"]
        if mn == "rhc":
            lbs = minim.get("parameters", {})
            max_oc = int(lbs.get("max_oracle_calls", 10000))
            minim["class"] = RHC_CLASS
            minim["parameters"] = OrderedDict([
                ("attributed", True),
                ("manip_attr", False),
                ("max_oracle_calls", max_oc),
                ("patience", 40),
                ("seed", int(seed)),
            ])
        else:
            minim["parameters"][MIN_SEED_KEY[mn]] = int(seed)
        if ds in FEATURE_BLIND:
            minim["parameters"]["recompute_features"] = False


def _delete_old_seeded(dry):
    n = 0
    for ds in DATASETS:
        for gen in GENERATORS:
            gdir = CFG_ROOT / ds / gen
            if not gdir.is_dir():
                continue
            for d in gdir.iterdir():
                if d.is_dir() and SEEDED_DIR_RE.match(d.name):
                    # keep dce-lcls-seed{1,2,3}; delete everything else seeded
                    if not (gen == "dce" and d.name.startswith("dce-lcls-seed")
                            and d.name.split("seed")[-1] in {"1", "2", "3"}):
                        if dry:
                            print(f"  rm {d.relative_to(REPO)}")
                        else:
                            import shutil; shutil.rmtree(d)
                        n += 1
    return n


def plan():
    """Yield (src, dst, ds, gen, mn, scope, seed)."""
    for ds in DATASETS:
        for gen in GENERATORS:
            for mn in MINIMIZERS:
                src = _src_dir(ds, gen, mn)
                if src is None or not src.is_dir():
                    print(f"  skip (no template): {ds}/{gen}/{gen}-{mn}", file=sys.stderr)
                    continue
                # no-seed canonical run (internal seed 0)
                dst = CFG_ROOT / ds / gen / f"{gen}-{mn}"
                yield src, dst, ds, gen, mn, f"{ds}_{gen}_{mn}", 0
                # extra LBS seeds: dce-lcls only
                if gen == "dce" and mn == "lcls":
                    for s in LBS_SEEDS:
                        dst_s = CFG_ROOT / ds / gen / f"dce-lcls-seed{s}"
                        yield src, dst_s, ds, gen, mn, f"{ds}_dce_lcls_seed{s}", s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("== Deleting superseded seeded E1 config dirs ==")
    n_del = _delete_old_seeded(args.dry_run)
    print(f"  {'would delete' if args.dry_run else 'deleted'} {n_del} dirs")

    print("== Generating configs (no-seed + dce-lcls seeds) ==")
    n = 0
    folds = list(range(10))
    for src, dst, ds, gen, mn, scope, seed in plan():
        for fold in folds:
            sf = src / f"generate_minimize{fold}.jsonc"
            if not sf.is_file():
                continue
            cfg = _read_jsonc(sf)
            _apply(cfg, ds, gen, mn, scope, seed)
            if args.dry_run:
                continue
            _write_json(dst / f"generate_minimize{fold}.jsonc", cfg)
            n += 1
    print(f"  {'would write' if args.dry_run else 'wrote'} {n} config files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
