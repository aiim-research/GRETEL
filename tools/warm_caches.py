#!/usr/bin/env python3
"""Build the cached objects of configs (dataset with its folds, oracle, explainer artefacts such as DCM
or the RSGG GANs) without running the experiment, so they can be committed before the configs are run
in parallel.

The dataset caches hold the random folds and are shared through git: if parallel jobs created them on
the server, each clone would end up with other folds. Each config runs in its own process (the Context
is a singleton) and stops right after the explainer is built. With ``--check`` nothing is created: it
lists what each config would load and whether it is there, stopping at the first missing object.

    python tools/warm_caches.py lab/config/generate_minimize/aids/dcm/dcm-lcls/generate_minimize0.jsonc
    python tools/warm_caches.py --check <config> [<config> ...]

Run it with CUDA hidden (``CUDA_VISIBLE_DEVICES=""``) so trained models hold no CUDA tensors, or re-save
them with ``tools/resave_cache_cpu.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MARK = "::WARM::"


def _child(cfg: str, check: bool) -> None:
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    from src.core import savable
    from src.utils.context import Context
    from src.evaluation.future.evaluator_manager_triplets import EvaluatorManager

    seen = []
    original = savable.Savable.load_or_create

    class Missing(Exception):
        pass

    def load_or_create(self, condition_ext=False):
        ctx = self.context
        qualifier = ctx.get_fullname(self).split('.')[1] + '_store_path'
        store = ctx._get_store_path(qualifier) or 'data'
        folder = os.path.join(store, str(self.dataset)) if 'dataset' in self.__dict__ else store
        path = os.path.normpath(os.path.join(folder, self.name))
        seen.append({"kind": qualifier[:-len('_store_path')], "path": path, "existed": os.path.exists(path)})
        if check and not seen[-1]["existed"]:
            raise Missing()
        return original(self, condition_ext)

    savable.Savable.load_or_create = load_or_create
    status = "ok"
    try:
        ctx = Context.get_context(cfg)
        ctx.run_number = 1
        EvaluatorManager(ctx)
    except Missing:
        status = "missing"
    except Exception as e:  # noqa: BLE001
        status = f"error {type(e).__name__}: {e}"
    print(MARK + json.dumps({"config": cfg, "status": status, "objects": seen}), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--check", action="store_true", help="create nothing, report what is missing")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        _child(args.configs[0], args.check)
        return 0
    failed = 0
    for cfg in args.configs:
        cmd = [sys.executable, __file__, "--child", cfg] + (["--check"] if args.check else [])
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO,
                             env={**os.environ, "GRETEL_LOG_FILE": "off"})
        line = next((l for l in out.stdout.splitlines() if l.startswith(MARK)), None)
        if line is None:
            print(f"{cfg}: crashed\n{out.stderr[-2000:]}")
            failed += 1
            continue
        r = json.loads(line[len(MARK):])
        failed += r["status"] not in ("ok", "missing")
        print(f"{cfg}: {r['status']}")
        for o in r["objects"]:
            state = "present" if o["existed"] else ("MISSING" if args.check else "created")
            print(f"    {state:<8} {o['kind']:<10} {o['path']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
