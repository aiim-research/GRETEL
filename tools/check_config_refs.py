#!/usr/bin/env python3
"""Static integrity check for every GRETEL configuration file.

Walks the repository, and for each ``.json`` / ``.jsonc`` config collects

  * every dotted ``src.*`` path used as a ``"class"`` reference, and
  * every relative path pointing at another repo file (``compose_*`` snippets,
    dataset directories, ...),

then verifies that the module/symbol and the file actually exist.  Run it
before and after any reorganisation: the counts must not grow.

Usage::

    python tools/check_config_refs.py                 # summary to stdout
    python tools/check_config_refs.py . report.json   # plus a JSON report

Exit code is 0 when the number of broken references did not grow relative to
``tools/config_refs_baseline.json`` (when that file exists), 1 otherwise.
"""
import ast, json, os, re, sys
from collections import defaultdict

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                       os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = sys.argv[2] if len(sys.argv) > 2 else None

CLS_RE = re.compile(r'"(src\.[A-Za-z0-9_.\-]+)"')
PATH_RE = re.compile(r'"((?:\./)?(?:lab|config|data|src|scripts|models|legacy|docs|tools)/[A-Za-z0-9_./\-\[\]+ ]+\.(?:json|jsonc|py|json5))"')

_symbols_cache = {}

def module_symbols(mod_file):
    if mod_file in _symbols_cache:
        return _symbols_cache[mod_file]
    syms = set()
    try:
        tree = ast.parse(open(mod_file, encoding="utf-8", errors="replace").read())
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                syms.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        syms.add(t.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    syms.add(a.asname or a.name.split(".")[0])
    except SyntaxError:
        syms = None
    _symbols_cache[mod_file] = syms
    return syms

def resolve(dotted):
    """Return (status, detail) for a dotted src.* path."""
    parts = dotted.split(".")
    # Longest matching module prefix (a .py file), else fall back to a package
    # __init__.py that re-exports the symbol.
    for cut in range(len(parts), 0, -1):
        base = os.path.join(ROOT, *parts[:cut])
        cand = None
        if os.path.isfile(base + ".py"):
            cand = base + ".py"
        elif os.path.isdir(base) and os.path.isfile(os.path.join(base, "__init__.py")):
            cand = os.path.join(base, "__init__.py")
        if cand is None:
            continue
        rest = parts[cut:]
        if not rest:
            return "MODULE_ONLY", cand
        syms = module_symbols(cand)
        if syms is None:
            return "OK_UNPARSED", cand
        if rest[0] in syms:
            return "OK", cand
        if cand.endswith("__init__.py"):
            # package prefix but symbol not re-exported: keep searching shorter
            continue
        return "MISSING_SYMBOL", f"{cand}::{'.'.join(rest)}"
    return "MISSING_MODULE", dotted

# Stored experiment output is not configuration: a results file records the
# class paths of a run that already happened, so a "broken" reference in one is
# history, not a defect. Scanning them would make every legitimate move of a
# retired module look like new damage.
SKIP_PREFIXES = ("lab/output/", "lab/output_legacy/")

# Configs that are kept for the record, not to be run. A broken reference here
# is documented history; one outside is a defect in the live tree.
RETIRED_PREFIXES = ("legacy/", "lab/config/legacy/")


def main():  # noqa: C901
    cfgs = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", "node_modules")]
        rel_dir = os.path.relpath(dirpath, ROOT).replace(os.sep, "/") + "/"
        if rel_dir.startswith(SKIP_PREFIXES):
            dirnames[:] = []
            continue
        for fn in filenames:
            if fn.endswith((".json", ".jsonc")):
                cfgs.append(os.path.join(dirpath, fn))

    report = {}
    bad_cls = defaultdict(list)
    bad_path = defaultdict(list)
    for c in cfgs:
        try:
            txt = open(c, encoding="utf-8", errors="replace").read()
        except Exception as e:
            report[os.path.relpath(c, ROOT)] = {"read_error": str(e)}
            continue
        classes = sorted(set(CLS_RE.findall(txt)))
        paths = sorted(set(PATH_RE.findall(txt)))
        cls_problems, path_problems = [], []
        for d in classes:
            st, detail = resolve(d)
            if st not in ("OK", "OK_UNPARSED", "MODULE_ONLY"):
                cls_problems.append([d, st])
                bad_cls[d].append(os.path.relpath(c, ROOT))
        for p in paths:
            pp = p[2:] if p.startswith("./") else p
            if not os.path.exists(os.path.join(ROOT, pp)):
                path_problems.append(p)
                bad_path[p].append(os.path.relpath(c, ROOT))
        rel = os.path.relpath(c, ROOT)
        report[rel] = {
            "classes": classes, "paths": paths,
            "bad_classes": cls_problems, "bad_paths": path_problems,
        }

    def retired(rel):
        return rel.startswith(RETIRED_PREFIXES)

    bad = [k for k, v in report.items() if v.get("bad_classes") or v.get("bad_paths")]
    bad_live = [k for k in bad if not retired(k)]
    print(f"configs scanned : {len(report)}")
    print(f"configs with problems: {len(bad)}"
          f"  ({len(bad_live)} in the live tree, {len(bad) - len(bad_live)} retired)")
    if bad_live:
        print("live-tree configs with problems:")
        for k in sorted(bad_live)[:20]:
            print(f"  {k}")
        if len(bad_live) > 20:
            print(f"  ... and {len(bad_live) - 20} more")
    print(f"\ndistinct unresolved class refs: {len(bad_cls)}")
    for d, files in sorted(bad_cls.items()):
        print(f"  {d}  ({len(files)} cfgs)  e.g. {files[0]}")
    print(f"\ndistinct missing file refs: {len(bad_path)}")
    for d, files in sorted(bad_path.items())[:60]:
        print(f"  {d}  ({len(files)} cfgs)  e.g. {files[0]}")
    if OUT:
        with open(OUT, "w") as f:
            json.dump(report, f, indent=1, sort_keys=True)

    baseline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "config_refs_baseline.json")
    if os.path.isfile(baseline_path):
        base = json.load(open(baseline_path))
        grew = []
        limit = base.get("live_tree_configs_with_problems")
        if limit is not None and len(bad_live) > limit:
            grew.append(f"live-tree configs {limit} -> {len(bad_live)}")
        if len(bad_cls) > base["unresolved_class_refs"]:
            grew.append(f"class refs {base['unresolved_class_refs']} -> {len(bad_cls)}")
        if len(bad_path) > base["missing_file_refs"]:
            grew.append(f"file refs {base['missing_file_refs']} -> {len(bad_path)}")
        if grew:
            print("\nFAIL: broken references grew: " + "; ".join(grew))
            return 1
        print(f"\nOK: no new broken references (baseline "
              f"{base['unresolved_class_refs']} class / "
              f"{base['missing_file_refs']} file / "
              f"{base.get('live_tree_configs_with_problems')} live-tree configs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
