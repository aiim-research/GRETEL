#!/usr/bin/env python3
"""Move Python modules inside ``src/`` and rewrite every reference to them.

A GRETEL module path is load-bearing in three different ways:

  * ``from src.a.b import C`` in Python source;
  * ``"class": "src.a.b.C"`` in a JSON/JSONC config;
  * ``"src.a.b.C"`` as a *string literal* in Python (``get_class``,
    ``set_proto_kls``, the ``init_dflts_to_of`` defaults).

The last two end up inside ``local_config``, and ``Context.get_name`` hashes
``local_config`` into every cache, artifact and result directory name.  Moving
a module that a live experiment references therefore silently invalidates
those caches.  Only move modules the reachability analysis marks as dead or
legacy-only (see ``tools/check_config_refs.py``).

Usage::

    python tools/move_module.py plan.tsv          # dry run
    python tools/move_module.py plan.tsv --apply

``plan.tsv`` holds one ``old_dotted<TAB>new_dotted`` pair per line; blank
lines and ``#`` comments are ignored.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_EXT = {".py", ".ipynb", ".json", ".jsonc", ".md", ".txt", ".sh", ".tex",
            ".yml", ".yaml", ".cfg", ".toml", ".in"}
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "node_modules"}


def read_plan(path):
    pairs = []
    for raw in open(path, encoding="utf-8"):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        old, new = line.split()
        pairs.append((old, new))
    return pairs


def dotted_to_file(dotted):
    return os.path.join(REPO, dotted.replace(".", os.sep) + ".py")


def ensure_pkg(directory):
    """Create the __init__.py chain under src/ so the new path imports."""
    created = []
    parts = os.path.relpath(directory, REPO).split(os.sep)
    if parts[0] != "src":
        return created
    for i in range(1, len(parts) + 1):
        d = os.path.join(REPO, *parts[:i])
        os.makedirs(d, exist_ok=True)
        init = os.path.join(d, "__init__.py")
        if not os.path.isfile(init):
            open(init, "w").close()
            created.append(os.path.relpath(init, REPO))
    return created


def text_files():
    for dp, dn, fn in os.walk(REPO):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for f in fn:
            if os.path.splitext(f)[1] in TEXT_EXT:
                yield os.path.join(dp, f)


def build_rules(pairs):
    """(regex, replacement) list, longest source first.

    Two spellings are rewritten: the dotted module path, and the slash path
    used in docs and shell commands (``src/a/b.py``).
    """
    rules = []
    for old, new in sorted(pairs, key=lambda p: -len(p[0])):
        rules.append((re.compile(r"(?<![A-Za-z0-9_.])" + re.escape(old)
                                 + r"(?![A-Za-z0-9_])"), new))
        rules.append((re.compile(r"(?<![A-Za-z0-9_/])" + re.escape(old.replace(".", "/"))
                                 + r"(?![A-Za-z0-9_/])"), new.replace(".", "/")))
    return rules


def rewrite(rules, apply):
    touched, hits = {}, 0
    for path in text_files():
        try:
            src = open(path, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        out, n = src, 0
        for rx, repl in rules:
            out, k = rx.subn(repl, out)
            n += k
        if n:
            touched[os.path.relpath(path, REPO)] = n
            hits += n
            if apply:
                open(path, "w", encoding="utf-8").write(out)
    return touched, hits


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    plan = read_plan(sys.argv[1])
    apply = "--apply" in sys.argv

    missing = [o for o, _ in plan if not os.path.isfile(dotted_to_file(o))]
    clashes = [n for _, n in plan if os.path.exists(dotted_to_file(n))]
    if missing or clashes:
        for m in missing:
            print("ERROR: source module not found:", m)
        for c in clashes:
            print("ERROR: destination already exists:", c)
        return 2

    touched, hits = rewrite(build_rules(plan), apply)
    print(f"{'APPLIED' if apply else 'DRY RUN'}: {len(plan)} module(s), "
          f"{hits} reference(s) in {len(touched)} file(s)")
    for f, n in sorted(touched.items(), key=lambda kv: (-kv[1], kv[0]))[:40]:
        print(f"  {n:6d}  {f}")
    if len(touched) > 40:
        print(f"  ... and {len(touched) - 40} more files")
    if not apply:
        return 0

    created = []
    for old, new in plan:
        dst = dotted_to_file(new)
        created += ensure_pkg(os.path.dirname(dst))
        subprocess.check_call(["git", "mv", dotted_to_file(old), dst], cwd=REPO)
    for c in created:
        subprocess.check_call(["git", "add", c], cwd=REPO)
    print(f"moved {len(plan)} file(s); created {len(created)} __init__.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
