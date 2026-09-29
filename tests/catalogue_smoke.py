#!/usr/bin/env python3
"""Smoke-test the published explanation methods GRETEL ships.

These are the comparison baselines the framework exists to offer: MEG, MACCS,
pRand, DDBS, CounteRGAN, EAGER, RSGG, COMBINEX and friends. Their
configurations live under ``legacy/config-v2/`` because that is the generation
they were published with, not because they are retired, and it is easy to let
them rot unnoticed since no current experiment exercises them.

For each entry the driver builds the dataset, the oracle and the explainer
through the real factories, exactly as ``EvaluatorManager`` does, then calls
``explain()`` on **one** instance and stops. That is the method running, not
merely importing.

Why subprocesses
----------------
``Context.get_context(cfg)`` caches the first Context on the class, so looping
over configs in one process would silently reuse the first dataset. One
subprocess per entry avoids it, the same way tests/regression_smoke.py does.

Usage from the repo root with the GRTL conda env active::

    python tests/catalogue_smoke.py
    python tests/catalogue_smoke.py --only meg maccs
    python tests/catalogue_smoke.py --timeout 900
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# name -> (config, index into the config's "explainers" list)
CATALOGUE = {
    "prand":      ("legacy/config-v2/TCR-150-100-0.3_TCC_pRand.jsonc", 0),
    "irand":      ("legacy/config-v2/ASD-ASDO/ASD_ASD-Custom_iRand.jsonc", 0),
    "dce":        ("legacy/config-v2/ASD-ASDO/ASD_ASD-Custom_DCE.jsonc", 0),
    "obs":        ("legacy/config-v2/ASD-ASDO/ASD_ASD-Custom_OBS.jsonc", 0),
    "ddbs":       ("legacy/config-v2/ASD-ASDO/ASD_ASD-Custom_DDBS.jsonc", 0),
    "maccs":      ("legacy/config-v2/BBBP-GCN/BBBP_GCN_MACCS.json", 0),
    "meg":        ("legacy/config-v2/leynier/meg-test-bbbp.json", 0),
    "rsgg":       ("legacy/config-v2/BBBP_GCN_RSGG.json", 0),
    "eager":      ("legacy/config-v2/EAGER/asd.json", 0),
    "gcountergan": ("legacy/config-v2/TCR-500-64-0.4_GCN_GCounteRGAN.jsonc", 0),
}

# Methods that already fail on main, before any reorganisation, so a run is
# only a regression when something NOT listed here breaks. Verified by an A/B
# against a worktree at main: identical error messages, same three methods.
# These are torch device-placement bugs (the model lands half on cuda and half
# on cpu), not packaging problems. Shrink this list, never grow it.
KNOWN_BROKEN = {
    "eager":       "device mismatch cuda:0/cpu, same on main",
    "gcountergan": "torch.cuda.FloatTensor vs torch.FloatTensor, same on main",
    "meg":         "device mismatch cuda:0/cpu, same on main",
}

# Missing optional dependencies are reported apart from real breakage.
OPTIONAL = ("No module named 'exmol'", "No module named 'selfies'",
            "No module named 'dgl'", "No module named 'omegaconf'",
            "No module named 'rdkit'", "No module named 'hnswlib'")


def _child(name: str) -> int:
    cfg_rel, idx = CATALOGUE[name]
    cfg = REPO / cfg_rel
    if not cfg.is_file():
        print(f"::RESULT:: status=MISSING_CFG detail={cfg_rel}", flush=True)
        return 2

    sys.path.insert(0, str(REPO))
    try:
        from src.utils.context import Context
        from src.dataset.dataset_factory import DatasetFactory
        from src.oracle.oracle_factory import OracleFactory
        from src.oracle.embedder_factory import EmbedderFactory
        from src.explainer.explainer_factory import ExplainerFactory

        ctx = Context.get_context(str(cfg))
        ctx.run_number = -1
        ctx.factories["datasets"] = DatasetFactory(ctx)
        ctx.factories["embedders"] = EmbedderFactory(ctx)
        ctx.factories["oracles"] = OracleFactory(ctx)
        ctx.factories["explainers"] = ExplainerFactory(ctx)

        pair = ctx.conf["do-pairs"][0]
        dataset = ctx.factories["datasets"].get_dataset(pair["dataset"])
        oracle = ctx.factories["oracles"].get_oracle(pair["oracle"], dataset)
        snippet = ctx.conf["explainers"][idx]
        kls = snippet["class"]
        explainer = ctx.factories["explainers"].get_explainer(snippet, dataset, oracle)

        instance = dataset.instances[0]
        result = explainer.explain(instance)
        kind = type(result).__name__
        print(f"::RESULT:: status=OK detail={kls} -> {kind}", flush=True)
        return 0
    except Exception as e:  # noqa: BLE001 - every failure is interesting here
        print(f"::RESULT:: status=FAIL detail={type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=sorted(CATALOGUE), metavar="NAME",
                    help="run just these methods (default: all)")
    ap.add_argument("--timeout", type=int, default=900,
                    help="per-method wall-clock budget in seconds (default 900)")
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        return _child(args.child)

    names = args.only or sorted(CATALOGUE)
    print(f"Running {len(names)} catalogue method(s), {args.timeout}s budget each.\n")
    rows, failed = [], 0
    for i, name in enumerate(names, 1):
        env = dict(os.environ, PYTHONWARNINGS="ignore")
        try:
            proc = subprocess.run(
                [sys.executable, __file__, "--child", name],
                cwd=REPO, env=env, capture_output=True, text=True, timeout=args.timeout)
            out = proc.stdout + proc.stderr
            line = next((l for l in out.splitlines() if l.startswith("::RESULT::")), "")
            status = line.split("status=")[1].split(" ")[0] if "status=" in line else "FAIL"
            detail = line.split("detail=", 1)[1] if "detail=" in line else out.strip()[-200:]
            if status == "FAIL" and any(o in out for o in OPTIONAL):
                status, detail = "SKIP", "optional dependency missing"
        except subprocess.TimeoutExpired:
            status, detail = "TIMEOUT", f"exceeded {args.timeout}s"
        if status in ("FAIL", "TIMEOUT") and name in KNOWN_BROKEN:
            status, detail = "KNOWN", f"{KNOWN_BROKEN[name]} [{detail[:60]}]"
        mark = {"OK": "ok  ", "SKIP": "skip", "TIMEOUT": "TIME",
                "MISSING_CFG": "cfg?", "KNOWN": "kno."}.get(status, "FAIL")
        failed += status in ("FAIL", "TIMEOUT")
        rows.append(f"  [{mark}] [{i:2d}/{len(names)}] {name:<12s} {detail}")
        print(rows[-1], flush=True)

    print(f"\nPassed: {sum('[ok  ]' in r for r in rows)}   "
          f"Skipped: {sum('[skip]' in r for r in rows)}   "
          f"Known-broken: {sum('[kno.]' in r for r in rows)}   "
          f"Failed: {failed}   Total: {len(rows)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
