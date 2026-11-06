#!/usr/bin/env python3
"""Build every missing selector checkpoint of the paper matrix from the collections that exist.

For each cell of ``gen_selector_paper_matrix.py`` and each collection fold (1 and 2), when
``lab/output/selector_pretrain/<cell>_<COLLECT_NAME>-f<c>.pt`` exists and
``models/pretrained/<CKPT_PREFIX>_<cell>-f<c>.pt`` does not, runs ``scripts/pretrain_selector_v3.py`` on it
(the selector settings come from the cell's selector_v1 config) and writes the checkpoint with its report
next to it. Idempotent: run it after every collection. The first sondeo's checkpoints
(``edge_selector_sel-pre_*``, from the ``_collect-f`` collections) are not touched.

    python scripts/pretrain_selector_checkpoints.py [--datasets bbbp-m70 tri-40 ...]

``--v2``: protocol v2 (no padding, ``lab/config/v2``): collections in ``lab/output/selector_pretrain/v2/``,
checkpoints in ``models/pretrained/v2/``, settings from the v2 selector_v1 configs (scripts/gen_v2_configs.py).
``--features`` (with ``--v2``): the arms with BLS-F substitutions, collections ``<cell>_collect2f-f<c>.pt`` and
checkpoints ``edge_selector_sel2f-pre_<cell>-f<c>.pt``, settings from the ``dce-lcls-sel-v1-f`` configs.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from gen_selector_paper_matrix import CKPT_PREFIX, COLLECT_FOLDS, COLLECT_NAME, SEL1, cells  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", default=cells())
    ap.add_argument("--v2", action="store_true", help="protocol v2 collections, checkpoints and configs")
    ap.add_argument("--features", action="store_true", help="the arms with BLS-F substitutions (protocol v2 only)")
    args = ap.parse_args()
    if args.features and not args.v2:
        ap.error("--features needs --v2")
    collect_name, ckpt_prefix, sel1 = COLLECT_NAME, CKPT_PREFIX, SEL1
    if args.features:
        collect_name, ckpt_prefix, sel1 = COLLECT_NAME + "f", CKPT_PREFIX.replace("sel2-pre", "sel2f-pre"), SEL1 + "-f"
    sub = "v2/" if args.v2 else ""
    tree = "lab/config/v2/generate_minimize" if args.v2 else "lab/config/generate_minimize"
    built, waiting = 0, []
    for ds in args.datasets:
        for c in COLLECT_FOLDS:
            examples = REPO / f"lab/output/selector_pretrain/{sub}{ds}_{collect_name}-f{c}.pt"
            out = REPO / f"models/pretrained/{sub}{ckpt_prefix}_{ds}-f{c}.pt"
            if out.exists():
                continue
            if not examples.exists():
                waiting.append(f"{ds} fold {c}")
                continue
            config = REPO / f"{tree}/{ds}-sel/dce/{sel1}/generate_minimize0.jsonc"
            out.parent.mkdir(parents=True, exist_ok=True)
            print(f"== {ds}, collection of fold {c} -> {out.relative_to(REPO)}")
            subprocess.run([sys.executable, str(REPO / "scripts/pretrain_selector_v3.py"), "--examples", str(examples),
                            "--config", str(config), "--out", str(out), "--report", str(out.with_suffix(".json"))],
                           check=True, cwd=REPO)
            built += 1
    print(f"built {built} checkpoint(s); collections still missing: {', '.join(waiting) if waiting else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
