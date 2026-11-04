"""Edit cost of a graph pair under an explicit node correspondence (protocol v2, task 0).

A counterfactual G' of a graph G is compared through a correspondence pi: a partial injective map from
the nodes of G to the nodes of G'. Nodes of G without an image are deleted, nodes of G' without a
preimage are inserted, an edge of G whose image is not an edge of G' is deleted (also when one of its
endpoints is deleted), an edge of G' that is not the image of an edge of G is inserted (also when one of
its endpoints is inserted), and a mapped node whose features change pays the distance between its two
feature vectors. The cost is

    d(G, G', pi) = c_node * (deleted + inserted nodes) + c_edge * (deleted + inserted edges)
                   + c_feat * sum over mapped v of ||x_v - x'_pi(v)||

and the graph edit distance is its minimum over pi. Padding plays no part: the two graphs keep their own
sizes, and what the padded comparison did implicitly (index i of G is index i of G', missing nodes are
empty rows) is the identity correspondence over the common indices.

For a counterfactual obtained by editing G (every minimizer, OFS, the random searches), pi is known: the
identity over the nodes the explainer kept, and d is the cost of its edit script. With pi the identity and
two graphs of the same size, the edge part equals the cell-wise count of ``ged.graph_edit_distance_metric``
(which never charges a node); unlike it, edges between common and extra nodes are charged when the sizes
differ, and a self-loop counts as one edit.

Properties (tests/edit_distance_unit.py): d >= 0, and d = 0 exactly when pi is a bijection that preserves
edges and features; d(G, G', pi) = d(G', G, pi^-1) with the inverse correspondence; adding one edit never
lowers d; and d(G, G'', pi2 . pi1) <= d(G, G', pi1) + d(G', G'', pi2).
"""
from __future__ import annotations

import numpy as np


def identity_correspondence(n1: int, n2: int) -> np.ndarray:
    """Node i of the first graph is node i of the second for every common index; the rest have no image."""
    mapping = np.full(n1, -1, dtype=int)
    common = min(n1, n2)
    mapping[:common] = np.arange(common)
    return mapping


def inverse_correspondence(mapping, n2: int) -> np.ndarray:
    inverse = np.full(n2, -1, dtype=int)
    for i, j in enumerate(np.asarray(mapping)):
        if j >= 0:
            inverse[j] = i
    return inverse


def compose_correspondences(first, second) -> np.ndarray:
    """second . first: node i goes to second[first[i]], or nowhere if either step drops it."""
    first, second = np.asarray(first), np.asarray(second)
    return np.array([second[j] if j >= 0 else -1 for j in first], dtype=int)


def _check(mapping, n1: int, n2: int) -> np.ndarray:
    mapping = np.asarray(mapping, dtype=int)
    if mapping.shape != (n1,):
        raise ValueError(f"The correspondence needs one entry per node of the first graph ({n1})")
    images = mapping[mapping >= 0]
    if (mapping < -1).any() or (images >= n2).any():
        raise ValueError("Correspondence entries are node indices of the second graph, or -1")
    if len(np.unique(images)) != len(images):
        raise ValueError("The correspondence must be injective")
    return mapping


def _edges(adj: np.ndarray, directed: bool) -> set[tuple[int, int]]:
    rows, cols = np.nonzero(adj)
    if directed:
        return {(int(u), int(v)) for u, v in zip(rows, cols)}
    return {(int(min(u, v)), int(max(u, v))) for u, v in zip(rows, cols)}


def edit_cost(adj1, adj2, features1=None, features2=None, mapping=None, directed: bool = False,
              node_cost: float = 1.0, edge_cost: float = 1.0, feature_cost: float = 1.0) -> dict:
    """The components and the total of d(G, G', pi); ``mapping`` defaults to the identity correspondence.

    Returns nodes_deleted, nodes_inserted, edges_deleted, edges_inserted, feature_distance (sum of the
    L2 distances of the mapped nodes' feature vectors; 0 when no features are given), structure
    (node and edge edits, unweighted) and total (with the costs)."""
    adj1, adj2 = np.asarray(adj1), np.asarray(adj2)
    n1, n2 = len(adj1), len(adj2)
    mapping = identity_correspondence(n1, n2) if mapping is None else _check(mapping, n1, n2)
    mapped = mapping >= 0
    nodes_deleted = int(n1 - mapped.sum())
    nodes_inserted = int(n2 - mapped.sum())

    edges1, edges2 = _edges(adj1, directed), _edges(adj2, directed)
    # Images of the edges whose two endpoints are mapped; the others are deleted with their endpoint.
    image = {(int(mapping[u]), int(mapping[v])) if directed
             else (int(min(mapping[u], mapping[v])), int(max(mapping[u], mapping[v])))
             for u, v in edges1 if mapping[u] >= 0 and mapping[v] >= 0}
    kept = len(image & edges2)
    edges_deleted = len(edges1) - kept
    edges_inserted = len(edges2) - kept

    feature_distance = 0.0
    if features1 is not None and features2 is not None and mapped.any():
        x1, x2 = np.asarray(features1, dtype=float), np.asarray(features2, dtype=float)
        if x1.shape[1:] != x2.shape[1:]:
            raise ValueError("The two graphs have different feature columns")
        rows = np.nonzero(mapped)[0]
        feature_distance = float(np.linalg.norm(x1[rows] - x2[mapping[rows]], axis=1).sum())

    structure = nodes_deleted + nodes_inserted + edges_deleted + edges_inserted
    total = (node_cost * (nodes_deleted + nodes_inserted) + edge_cost * (edges_deleted + edges_inserted)
             + feature_cost * feature_distance)
    return {"nodes_deleted": nodes_deleted, "nodes_inserted": nodes_inserted,
            "edges_deleted": edges_deleted, "edges_inserted": edges_inserted,
            "feature_distance": feature_distance, "structure": structure, "total": float(total)}


def instance_edit_cost(instance, counterfactual, mapping=None, **costs) -> dict:
    """edit_cost for two GraphInstance objects."""
    return edit_cost(instance.data, counterfactual.data, instance.node_features, counterfactual.node_features,
                     mapping, instance.directed and counterfactual.directed, **costs)
