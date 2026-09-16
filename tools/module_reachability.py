#!/usr/bin/env python3
"""Classify every module under ``src/`` as ACTIVE, LEGACY-ONLY or ORPHAN.

A module is ACTIVE when the *current* experiment surface reaches it: the
config trees the revision batch runs, the entry points, ``scripts/``,
``tests/`` and the live notebooks.  It is LEGACY-ONLY when nothing but the
retired configs and notebooks reach it, and ORPHAN when nothing reaches it at
all.

Reachability follows three kinds of edge, because GRETEL resolves classes both
statically and by name:

  * ``import`` / ``from ... import`` statements,
  * ``"class": "src.a.b.C"`` entries in configs,
  * dotted ``src.*`` **string literals** inside Python (``get_class``,
    ``set_proto_kls``, ``init_dflts_to_of`` defaults).

That third kind is easy to miss and expensive to get wrong: those strings land
in ``local_config``, which ``Context.get_name`` hashes into cache and result
directory names.  Anything ACTIVE must therefore keep its module path.

Usage::

    python tools/module_reachability.py [report.json]
"""
import ast, os, re, json, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLS_RE = re.compile(r'"(src\.[A-Za-z0-9_.\-]+)"')

# --- active config trees (current revision work) -------------------------
ACTIVE_CFG_DIRS = ["lab/config/generate_minimize", "lab/config/tagging",
                   "lab/config/bls_selection_net", "lab/config/bls_selection_net_trainable",
                   "lab/config/meta", "lab/config/meta_ens", "lab/config/metaheuristics",
                   "lab/config/ensembles", "lab/config/llm_exp_generate_minimize",
                   "lab/config/debug", "lab/config/base", "lab/config/snippets"]
ACTIVE_PY = ["main.py", "future_main.py", "scripts", "tests"]
ACTIVE_NB = ["lab/notebooks"]   # lab/notebooks/legacy/ is deliberately excluded

def all_src_modules():
    mods = set()
    for dp, dn, fn in os.walk(os.path.join(ROOT, "src")):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in fn:
            if f.endswith(".py"):
                rel = os.path.relpath(os.path.join(dp, f), ROOT)
                mods.add(rel[:-3].replace(os.sep, "."))
    return mods
MODS = all_src_modules()

def mod_file(m):
    p = os.path.join(ROOT, m.replace(".", os.sep) + ".py")
    return p if os.path.isfile(p) else None

DOTTED = re.compile(r"^src\.[A-Za-z0-9_.]+$")


def imports_of(path):
    """Import edges AND dotted src.* strings passed to get_class() at runtime.

    The string literals matter as much as the imports: they end up in
    ``local_config`` defaults, which are hashed into every cache and result
    name. They are read off the AST rather than the raw text, so a path that
    only survives in a commented-out line does not keep a dead module alive.
    """
    out = set()
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("src."):
                    out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("src") and node.level == 0:
                out.add(node.module)
                for a in node.names:
                    out.add(node.module + "." + a.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if DOTTED.match(node.value):
                out.add(node.value)
    return out


def normalize(d):
    parts = d.split(".")
    for cut in range(len(parts), 0, -1):
        c = ".".join(parts[:cut])
        if c in MODS: return c
        if c + ".__init__" in MODS: return c + ".__init__"
    return None

def walk(seeds):
    seen, stack = set(), [n for n in (normalize(s) for s in seeds) if n]
    while stack:
        m = stack.pop()
        if m in seen: continue
        seen.add(m)
        f = mod_file(m)
        if not f: continue
        for i in imports_of(f):
            n = normalize(i)
            if n and n not in seen: stack.append(n)
    return seen

def cfg_refs(paths):
    refs = set()
    for p in paths:
        fp = os.path.join(ROOT, p)
        files = []
        if os.path.isfile(fp): files = [fp]
        else:
            for dp, dn, fn in os.walk(fp):
                files += [os.path.join(dp, f) for f in fn if f.endswith((".json", ".jsonc"))]
        for f in files:
            refs |= set(CLS_RE.findall(open(f, encoding="utf-8", errors="replace").read()))
    return refs

def py_refs(paths):
    seeds = set()
    for p in paths:
        fp = os.path.join(ROOT, p)
        if os.path.isfile(fp): seeds |= imports_of(fp)
        elif os.path.isdir(fp):
            for dp, dn, fn in os.walk(fp):
                dn[:] = [x for x in dn if x != "__pycache__"]
                for f in fn:
                    if f.endswith(".py"): seeds |= imports_of(os.path.join(dp, f))
    return seeds

def nb_refs(paths):
    """src.* references in notebook *code cells*.

    Stored outputs are run logs: they name classes a past run instantiated,
    which says nothing about whether today's code still reaches them.
    """
    files = []
    for p in paths:
        fp = os.path.join(ROOT, p)
        if os.path.isfile(fp):
            files.append(fp)
        elif os.path.isdir(fp):
            files += [os.path.join(fp, f) for f in sorted(os.listdir(fp))
                      if f.endswith(".ipynb")]
    refs = set()
    for fp in files:
        try:
            nb = json.load(open(fp, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            refs |= set(re.findall(r"src\.[A-Za-z0-9_.]+", src))
    return refs


active_seeds = cfg_refs(ACTIVE_CFG_DIRS) | py_refs(ACTIVE_PY) | nb_refs(ACTIVE_NB)
ACTIVE = walk(active_seeds)

# everything else that still references src.*
ALL_CFG = cfg_refs(["config", "lab/config", "legacy"])
ALL_NB = set()
for dp, dn, fn in os.walk(ROOT):
    dn[:] = [d for d in dn if d not in (".git", "__pycache__")]
    for f in fn:
        if f.endswith(".ipynb"):
            ALL_NB |= nb_refs([os.path.relpath(os.path.join(dp, f), ROOT)])
OTHER = walk(ALL_CFG | ALL_NB | py_refs(["legacy"])) - ACTIVE
ORPHAN = sorted(m for m in MODS if m not in ACTIVE and m not in OTHER)

out = {"active": sorted(ACTIVE), "legacy_only": sorted(OTHER), "orphan": ORPHAN}
print(f"ACTIVE      : {len(ACTIVE):4d}  (of which __init__: {sum(1 for m in ACTIVE if m.endswith('__init__'))})")
print(f"LEGACY-ONLY : {len(OTHER):4d}")
print(f"ORPHAN      : {len(ORPHAN):4d}  (of which __init__: {sum(1 for m in ORPHAN if m.endswith('__init__'))})")
print("\n=== LEGACY-ONLY ===")
for m in sorted(OTHER): print("  ", m)
print("\n=== ORPHAN (non __init__) ===")
for m in ORPHAN:
    if not m.endswith("__init__"): print("  ", m)
if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], "w"), indent=1)
