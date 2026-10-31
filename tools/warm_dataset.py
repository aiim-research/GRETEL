#!/usr/bin/env python3
"""Build (or load) only the dataset cache of each config, without its oracle or explainer: for datasets
whose cache has to be regenerated while their explainer caches are expensive (RSGG, DCM). Each config
runs in its own process (the Context is a singleton).

    python tools/warm_dataset.py <config> [<config> ...]
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _child(cfg: str) -> None:
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    from src.utils.context import Context
    from src.dataset.dataset_factory import DatasetFactory
    ctx = Context.get_context(cfg)
    ctx.factories['datasets'] = DatasetFactory(ctx)
    ctx.run_number = 1
    triplet = ctx.conf['doe-triplets'][0]
    dataset = ctx.factories['datasets'].get_dataset(triplet['dataset'])
    print(f"{cfg}: {ctx.get_path(dataset)} ({len(dataset.instances)} instances, {len(dataset.splits)} folds)", flush=True)


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        _child(sys.argv[2])
        return 0
    failed = 0
    for cfg in sys.argv[1:]:
        out = subprocess.run([sys.executable, __file__, "--child", cfg], capture_output=True, text=True, cwd=REPO,
                             env={**os.environ, "GRETEL_LOG_FILE": "off"})
        lines = [l for l in out.stdout.splitlines() if l.startswith(cfg)]
        print(lines[-1] if lines else f"{cfg}: failed\n{out.stderr[-1500:]}")
        failed += not lines
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
