#!/usr/bin/env python3
"""Re-save cached trained models (oracles, mainly) with every tensor on the CPU.

A model trained on a GPU is pickled with CUDA tensors, and unpickling it on a machine without CUDA
fails. The caches under ``lab/data/cache`` are shared through git, so they must load anywhere: this
script loads each pickle (on a machine with CUDA), moves the model to the CPU, writes it back in place,
and then checks in a fresh process with CUDA hidden that the file loads and that every parameter and
buffer is bit-identical to the original. Files already on the CPU are left untouched.

    python tools/resave_cache_cpu.py                       # every cached oracle
    python tools/resave_cache_cpu.py lab/data/cache/oracles/AIDS-*/OracleTorch-*
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _modules(model):
    import torch
    items = model if isinstance(model, (list, tuple)) else [model]
    return [m for m in items if isinstance(m, torch.nn.Module)]


def _digest(model) -> str:
    """Hash of every parameter and buffer, taken on the CPU (device-independent)."""
    h = hashlib.sha256()
    for m in _modules(model):
        for name, t in sorted(m.state_dict().items()):
            h.update(name.encode())
            h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _on_cuda(model) -> bool:
    return any(t.is_cuda for m in _modules(model) for t in m.state_dict().values())


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="*", help="cache files (default: lab/data/cache/oracles/*/*)")
    ap.add_argument("--digest", help=argparse.SUPPRESS)  # internal: print the digest of one file
    args = ap.parse_args()

    if args.digest:
        print(_digest(_load(args.digest)["model"]))
        return 0

    files = args.files or sorted(glob.glob(str(REPO / "lab/data/cache/oracles/*/*")))
    files = [f for f in files if not f.endswith((".lck", ".lock"))]
    failed = 0
    for f in files:
        dump = _load(f)
        if not isinstance(dump, dict) or "model" not in dump:
            print(f"skip (not a model dump)  {f}")
            continue
        if not _on_cuda(dump["model"]):
            print(f"cpu already              {f}")
            continue
        before = _digest(dump["model"])
        for m in _modules(dump["model"]):
            m.to("cpu")
        with open(f, "wb") as out:
            pickle.dump(dump, out)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        after = subprocess.run([sys.executable, __file__, "--digest", f], env=env, cwd=REPO,
                               capture_output=True, text=True)
        ok = after.returncode == 0 and after.stdout.strip() == before
        failed += not ok
        print(f"{'re-saved on cpu' if ok else 'MISMATCH':<24} {f}")
        if not ok:
            print(after.stderr[-2000:])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
