"""Isolated-node policy of protocol v2: a node left without edges is deleted from the graph.

Datasets whose graphs are connected by nature (molecules, proteins, ...) declare it with the manipulator
``src.dataset.manipulators.isolated.DeleteIsolatedNodes``, which also removes the nodes that are already
isolated when the dataset loads (salt ions in molecules). Explainers keep editing in the input's frame
(same node indices); the oracle drops the nodes an edit left isolated before predicting, and the
counterfactual is materialized without them, its node correspondence recording them as deleted.
Datasets where isolated nodes are part of the data (COLORS-3, Synthie) and fixed-node ones (ASD) do not
declare it, and nothing changes for them.
"""
from __future__ import annotations

import copy

import numpy as np


def isolated_nodes(adj) -> np.ndarray:
    adj = np.asarray(adj)
    if not len(adj):
        return np.zeros(0, dtype=int)
    return np.nonzero(~((adj != 0).any(0) | (adj != 0).any(1)))[0]


def drop_isolated(graph):
    """(graph without its isolated nodes, kept node indices). The graph itself when it has none, or when
    every node is isolated (an empty graph is no counterfactual)."""
    gone = isolated_nodes(graph.data)
    n = len(graph.data)
    if len(gone) == 0 or len(gone) == n:
        return graph, np.arange(n)
    keep = np.setdiff1d(np.arange(n), gone)
    out = copy.copy(graph)
    out.data = np.asarray(graph.data)[np.ix_(keep, keep)]
    out.node_features = np.asarray(graph.node_features)[keep]
    # Edge attributes follow the row-major order of the non-zero cells, which removing empty rows and
    # columns does not change.
    out._nx_repr = None
    return out, keep


def materialize(graph, policy: str):
    """(graph as the oracle sees it, correspondence from its frame to the result: -1 for deleted nodes)."""
    n = len(graph.data)
    if policy != "delete":
        return graph, np.arange(n)
    out, keep = drop_isolated(graph)
    mapping = np.full(n, -1, dtype=int)
    mapping[keep] = np.arange(len(keep))
    return out, mapping


def policy_of(dataset) -> str:
    return getattr(dataset, "isolated_nodes", "keep") if dataset is not None else "keep"
