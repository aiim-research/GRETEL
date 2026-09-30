#!/usr/bin/env python3
"""Build a config's dataset, oracle and explainer caches, then list the test instances of one fold.

Writes one row per test instance (id, nodes, edges, label, oracle prediction) so a fixed random sample
can be drawn for the selector study. Running it also warms the dataset and oracle caches, which on a
new machine or dataset take minutes and would otherwise be paid by the first experiment.

    python scripts/list_fold_instances.py lab/config/generate_minimize/aids/dce/dce-lcls/generate_minimize0.jsonc \\
        lab/output/fold_instances/aids_fold0.json [--fold 0]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import torch  # noqa: F401,E402  (before the framework, for its BLAS side effects)
from src.utils.context import Context  # noqa: E402
from src.evaluation.future.evaluator_manager_triplets import EvaluatorManager  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config"); ap.add_argument("out"); ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    ctx = Context.get_context(args.config)
    ctx.run_number = 1
    ev = EvaluatorManager(ctx).evaluators[0]
    ds, orc = ev.dataset, ev.oracle
    test = set(ds.splits[args.fold]["test"])
    rows = [{"id": int(i.id), "nodes": int(i.num_nodes), "edges": int(i.num_edges), "label": int(i.label),
             "pred": int(orc.predict(i))} for i in ds.instances if i.id in test]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=1)
    print(f"{ds.name}: fold {args.fold} test size {len(rows)}, oracle correct {sum(r['label'] == r['pred'] for r in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
