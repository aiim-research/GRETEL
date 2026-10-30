#!/usr/bin/env python3
"""Build every missing selector checkpoint of the paper matrix from the collections that exist.

For each dataset of ``gen_selector_paper_matrix.py`` and each collection fold (1 and 2), when
``lab/output/selector_pretrain/<dataset>_collect-f<c>.pt`` exists and
``models/pretrained/edge_selector_sel-pre_<dataset>-f<c>.pt`` does not, runs
``scripts/pretrain_selector_v3.py`` on it (the selector settings come from the dataset's selector_v1
config) and writes the checkpoint with its report next to it. Idempotent: run it after every collection.

    python scripts/pretrain_selector_checkpoints.py [--datasets asd bbbp ...]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from gen_selector_paper_matrix import COLLECT_FOLDS, DATASETS, SUBSETS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", default=DATASETS + list(SUBSETS))
    args = ap.parse_args()
    built, waiting = 0, []
    for ds in args.datasets:
        for c in COLLECT_FOLDS:
            examples = REPO / f"lab/output/selector_pretrain/{ds}_collect-f{c}.pt"
            out = REPO / f"models/pretrained/edge_selector_sel-pre_{ds}-f{c}.pt"
            if out.exists():
                continue
            if not examples.exists():
                waiting.append(f"{ds} fold {c}")
                continue
            config = REPO / f"lab/config/generate_minimize/{ds}-sel/dce/dce-lcls-sel/generate_minimize0.jsonc"
            print(f"== {ds}, collection of fold {c} -> {out.relative_to(REPO)}")
            subprocess.run([sys.executable, str(REPO / "scripts/pretrain_selector_v3.py"), "--examples", str(examples),
                            "--config", str(config), "--out", str(out), "--report", str(out.with_suffix(".json"))],
                           check=True, cwd=REPO)
            built += 1
    print(f"built {built} checkpoint(s); collections still missing: {', '.join(waiting) if waiting else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
