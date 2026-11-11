"""Prototype (11 November 2026): rebuild the node correspondence of v2 dumps written before the evaluator stored
it (``cf_per_instance/fold_*/cf_*.json`` without a "correspondence" key), so a later measure (line B1) can be
computed on past runs without running them again.

A minimizer edits in the input's frame and the isolated-node policy then deletes the nodes left without edges,
keeping the order, so the correspondence is a monotone map of the counterfactual's nodes into the input's. The
search enumerates those maps by branch and bound and keeps the ones whose structural and feature cost equal the
stored StructuralEditDistance and MatchedFeatureDistance (the stored metrics act as a checksum).

First run on six scopes (AIDS selector and bls_v0, BBBP RHC-B, BZR OBS, PROTEINS and ENZYMES bls_v1-F): a unique
correspondence for 85 to 100 percent of the instances. Not solved here: fallback seeds (the generator's
candidate itself, not a monotone map; their correspondence is in the generator cache, ``generator_outputs``),
searches that hit the step limit, and a few instances with no or several matching maps.

    python tools/rebuild_correspondences.py lab/output/results/<scope>
"""
import json, glob, sys, pickle, collections, os
import numpy as np
sys.path.insert(0, "/home/rodrigo/projects/GRETEL")
from src.utils.metrics.correspondence import edit_cost
sys.setrecursionlimit(20000)

def adj(g):
    n = g["num_nodes"]; A = np.zeros((n, n), bool)
    for i, j in g["edges"]: A[i, j] = A[j, i] = True
    return A
def feats(g): return None if g["node_features"] is None else np.asarray(g["node_features"], float)

def solve(G, C, S, F, limit=300000):
    n, k = G["num_nodes"], C["num_nodes"]
    if k > n: return None
    AG, AC = adj(G), adj(C); fG, fC = feats(G), feats(C)
    sols, steps = [], [0]
    m = np.full(n, -1)
    # cost so far: deleted nodes, input edges touching a deleted node, edge mismatches among mapped pairs
    def rec(i, j, s, f, mapped, dele):
        steps[0] += 1
        if steps[0] > limit or len(sols) > 2: return
        if s > S + 1e-9 or f > F + 1e-4: return
        if j == k:
            rest = range(i, n); s2 = s + len(rest)
            dl = dele | set(rest)
            for d in rest:
                s2 += sum(1 for v in np.nonzero(AG[d])[0] if v not in dl or v > d)
            # CF edges among mapped nodes already counted; CF edges are all between mapped nodes
            if abs(s2 - S) < 1e-9 and abs(f - F) < 1e-4:
                mm = m.copy(); mm[list(rest)] = -1; sols.append(mm)
            return
        if n - i < k - j: return
        # map input i -> cf j
        ds = sum(1 for (i2, j2) in mapped if AC[j, j2] != AG[i, i2])
        ds += sum(1 for d in dele if AG[i, d])          # edge to an earlier deleted node: deleted with it
        df = 0.0 if fG is None else float(np.linalg.norm(fG[i] - fC[j]))
        m[i] = j; rec(i + 1, j + 1, s + ds, f + df, mapped + [(i, j)], dele); m[i] = -1
        if n - i - 1 >= k - j:
            ds = 1 + sum(1 for (i2, _) in mapped if AG[i, i2]) + sum(1 for d in dele if AG[i, d])
            rec(i + 1, j, s + ds, f, mapped, dele | {i})
    rec(0, 0, 0, 0.0, [], frozenset())
    return sols if steps[0] <= limit else (sols or None)

def check(G, C, m, met):
    r = edit_cost(adj(G).astype(int), adj(C).astype(int), feats(G), feats(C), m)
    return abs(r["structure"] - met["StructuralEditDistance"]) < 1e-6 and abs(r["feature_distance"] - met["MatchedFeatureDistance"]) < 1e-4

st = collections.Counter(); amb_same = collections.Counter()
for f in glob.glob(sys.argv[1] + "/**/cf_per_instance/fold_*/cf_*.json", recursive=True):
    d = json.load(open(f)); G, C, met = d["input"], d["counterfactual"], d["metrics"]
    if C is None or met.get("Correctness", 0) < 1: st["no cf"] += 1; continue
    sols = solve(G, C, met["StructuralEditDistance"], met.get("MatchedFeatureDistance", 0.0))
    if sols is None: st["gave up / larger cf (fallback)"] += 1; continue
    sols = [s for s in sols if check(G, C, s, met)]
    if len(sols) == 1: st["unique"] += 1
    elif not sols: st["none"] += 1
    else:
        st["ambiguous"] += 1
        # does the ambiguity matter for B1? same deleted set's feature norms?
        fG = feats(G)
        norms = {round(float(sum(np.linalg.norm(fG[i]) for i in np.nonzero(s < 0)[0])) if fG is not None else 0, 6) for s in sols}
        amb_same["same deleted norm" if len(norms) == 1 else "different"] += 1
print(dict(st), dict(amb_same))
