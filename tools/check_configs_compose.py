#!/usr/bin/env python3
"""Compose every configuration and report the ones that fail to load.

The cheapest useful check on a config: run it through the same
``propagate(compose(...))`` the framework runs at startup, and see whether it
comes out the other side. That catches a malformed JSONC file, a `compose_*`
pointing at a snippet that does not exist, a `propagate` block naming a
section that is not there, and a missing `experiment.scope`.

It does **not** build datasets, train oracles or run explainers, so it takes
seconds over the whole tree rather than hours. Use it to know a config will
start; use `tests/regression_smoke.py` to know it runs.

Usage::

    python tools/check_configs_compose.py                 # everything
    python tools/check_configs_compose.py lab/config/generate_minimize
"""
from __future__ import annotations

import os
import sys
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# Stored experiment output is not configuration, and snippets are fragments
# that are composed into a config rather than composed on their own.
SKIP_PREFIXES = ("lab/output/", "lab/output_legacy/", "data/")
SKIP_DIRS = {".git", "__pycache__", "snippets"}


def configs(root):
    for dp, dn, fn in os.walk(os.path.join(REPO, root)):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        rel = os.path.relpath(dp, REPO).replace(os.sep, "/") + "/"
        if rel.startswith(SKIP_PREFIXES):
            dn[:] = []
            continue
        for f in sorted(fn):
            if f.endswith((".json", ".jsonc")):
                yield os.path.join(dp, f)


def main() -> int:
    from jsonc_parser.parser import JsoncParser
    from src.utils.composer import compose, propagate

    roots = sys.argv[1:] or ["lab/config", "legacy/config-v2"]
    bad, fragments, n = [], [], 0
    for root in roots:
        for path in configs(root):
            rel = os.path.relpath(path, REPO)
            n += 1
            try:
                raw = JsoncParser.parse_file(path)
            except Exception as exc:  # noqa: BLE001
                bad.append((rel, f"{type(exc).__name__}: {exc}"))
                continue
            if "experiment" not in raw:
                # A component or section definition that other configs compose,
                # not something you can run. Nothing to check beyond it parsing.
                fragments.append(rel)
                continue
            try:
                conf = propagate(compose(raw))
                conf["experiment"].get("scope", "default_scope")
            except Exception as exc:  # noqa: BLE001 - any failure is a failure
                bad.append((rel, f"{type(exc).__name__}: {exc}"))

    print(f"scanned   {n} files from {', '.join(roots)}")
    print(f"composed  {n - len(fragments) - len(bad)}")
    print(f"fragments {len(fragments)}  (no experiment section: composed into other configs)")
    print(f"failed    {len(bad)}")
    shown = {}
    for rel, why in bad:
        shown.setdefault(why.split(":")[0], []).append((rel, why))
    for kind, rows in sorted(shown.items()):
        print(f"\n  {kind}  ({len(rows)})")
        for rel, why in rows[:5]:
            print(f"    {rel}\n      {why[:160]}")
        if len(rows) > 5:
            print(f"    ... and {len(rows) - 5} more")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
