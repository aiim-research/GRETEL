"""Projection of a candidate graph onto the nodes of an input (protocol v2, the generator contract).

Generators that return another graph of the dataset (DCE, DCM) give the minimizer a seed with other nodes
than the input. In v1 the padding hid that: the seed's extra nodes landed on the input's padding rows and
the minimizer edited them as if they existed. In v2 the seed is first projected onto the input: a
correspondence pi maps input nodes to candidate nodes, every pair of mapped input nodes takes the
candidate's adjacency between their images, and input nodes without an image lose their edges (under the
isolated-node policy "delete" they are then deleted: with a candidate no larger than the input, the full
projection is the candidate itself). The generator returns the nearest candidate whose projection the
minimizer accepts as a counterfactual, so the minimizer always starts from a valid seed inside its domain;
when there is none it falls back to the nearest candidate itself, which the minimizer cannot shrink.

Correspondences (``correspondence``):

* ``index``: input node i is candidate node i for every common index (what the padded comparison did);
* ``assignment``: a minimum-cost assignment between the two node sets (Hungarian algorithm) on the feature
  distance plus the degree difference, so atoms are matched with atoms of the same element where possible.
"""
from __future__ import annotations

import copy

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.utils.metrics.correspondence import edit_cost, identity_correspondence

CORRESPONDENCES = ("index", "assignment")


def assignment_correspondence(instance, candidate) -> np.ndarray:
    """Input node -> candidate node (or -1), minimising feature distance plus degree difference."""
    x1, x2 = np.asarray(instance.node_features, dtype=float), np.asarray(candidate.node_features, dtype=float)
    d1, d2 = (np.asarray(instance.data) != 0).sum(1), (np.asarray(candidate.data) != 0).sum(1)
    if x1.shape[1:] == x2.shape[1:] and x1.size and x2.size:
        cost = np.linalg.norm(x1[:, None, :] - x2[None, :, :], axis=2)
    else:
        cost = np.zeros((len(x1), len(x2)))
    cost = cost + np.abs(d1[:, None] - d2[None, :])
    rows, cols = linear_sum_assignment(cost)
    mapping = np.full(len(x1), -1, dtype=int)
    mapping[rows] = cols
    return mapping


def correspondence(instance, candidate, kind: str) -> np.ndarray:
    if kind == "index":
        return identity_correspondence(len(instance.data), len(candidate.data))
    if kind == "assignment":
        return assignment_correspondence(instance, candidate)
    raise ValueError(f"Unknown correspondence {kind!r}; expected one of {CORRESPONDENCES}")


def _edge_attributes(graph):
    return {tuple(map(int, coord)): (graph.edge_weights[i], graph.edge_features[i])
            for i, coord in enumerate(zip(*np.nonzero(graph.data)))}


def project(instance, candidate, mapping, features: bool = False):
    """The candidate transported onto the input's nodes, in the input's frame (same node indices).

    Mapped pairs take the candidate's adjacency and the candidate's edge attributes; input nodes without
    an image lose all their edges (under the policy "delete" they are then deleted, so with every candidate
    node mapped the result is the candidate itself, up to node names). Node features stay the input's
    (``features=False``, what the edge-only minimizers keep) or become the candidate's on mapped nodes
    (``features=True``, the full projection)."""
    mapping = np.asarray(mapping, dtype=int)
    rows = np.nonzero(mapping >= 0)[0]
    data = np.zeros_like(np.asarray(instance.data, dtype=float))
    data[np.ix_(rows, rows)] = np.asarray(candidate.data)[np.ix_(mapping[rows], mapping[rows])]
    theirs = _edge_attributes(candidate)
    width = instance.edge_features.shape[1] if np.ndim(instance.edge_features) == 2 else 0
    weights, edge_features = [], []
    for u, v in zip(*np.nonzero(data)):
        weight, attrs = theirs[(int(mapping[u]), int(mapping[v]))]
        weights.append(weight)
        edge_features.append(np.asarray(attrs, dtype=float).reshape(-1)[:width] if width else np.zeros(0))
    projected = copy.deepcopy(instance)
    projected.data = data
    projected.edge_weights = np.asarray(weights, dtype=np.float32)
    projected.edge_features = np.asarray(edge_features, dtype=np.float32).reshape(len(weights), width)
    if features:
        x = np.array(instance.node_features, dtype=float, copy=True)
        y = np.asarray(candidate.node_features, dtype=float)
        if x.shape[1:] == y.shape[1:]:
            x[rows] = y[mapping[rows]]
        projected.node_features = x
    projected._nx_repr = None
    return projected


def structural_distance(instance, candidate, mapping) -> float:
    """Node and edge edits between the two graphs under the correspondence (the v2 measure's structure)."""
    return edit_cost(instance.data, candidate.data, mapping=mapping,
                     directed=instance.directed and candidate.directed)["structure"]


def ranked_candidates(instance, candidates, kind: str):
    """[(structural distance under the correspondence, order, candidate, correspondence)], nearest first."""
    ranked = []
    for candidate in candidates:
        mapping = correspondence(instance, candidate, kind)
        ranked.append((structural_distance(instance, candidate, mapping), len(ranked), candidate, mapping))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked


def nearest_valid_projection(instance, candidates, is_valid, kind: str, features: bool = False):
    """(projection, candidate, correspondence) for the nearest candidate whose projection ``is_valid``
    accepts, or (None, None, None). ``is_valid(graph)`` is the minimizer's own test (the oracle, plus its
    attribute methods when it uses them); each test spends oracle calls, counted as the generator's."""
    for _, _, candidate, mapping in ranked_candidates(instance, candidates, kind):
        projected = project(instance, candidate, mapping, features)
        if np.array_equal(projected.data, instance.data) and (
                not features or np.array_equal(projected.node_features, instance.node_features)):
            continue   # nothing of the candidate lands on the input's nodes
        if is_valid(projected):
            return projected, candidate, mapping
    return None, None, None
