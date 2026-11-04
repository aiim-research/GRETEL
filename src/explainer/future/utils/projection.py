"""Projection of a candidate graph onto the nodes of an input (protocol v2, the generator contract).

Generators that return another graph of the dataset (DCE, DCM) give the minimizer a seed with other nodes
than the input. In v1 the padding hid that: the seed's extra nodes landed on the input's padding rows and
the minimizer edited them as if they existed. In v2 the seed is first projected onto the input: a
correspondence pi maps input nodes to candidate nodes, and every pair of mapped input nodes takes the
candidate's adjacency value between their images. The projection keeps the input's nodes and node features
(what the minimizers keep and evaluate, without attribute manipulation), edges present in the input keep
their attributes, and new edges take the candidate's. The generator returns the nearest candidate whose
projection is a counterfactual, so the minimizer always starts from a valid seed inside its domain.

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


def project(instance, candidate, mapping):
    """The input's nodes and features with the candidate's structure among the mapped nodes."""
    mapping = np.asarray(mapping, dtype=int)
    rows = np.nonzero(mapping >= 0)[0]
    data = np.array(instance.data, copy=True)
    data[np.ix_(rows, rows)] = np.asarray(candidate.data)[np.ix_(mapping[rows], mapping[rows])]
    own, theirs = _edge_attributes(instance), _edge_attributes(candidate)
    width = instance.edge_features.shape[1] if np.ndim(instance.edge_features) == 2 else 0
    weights, features = [], []
    for u, v in zip(*np.nonzero(data)):
        key = (int(u), int(v))
        if key in own:
            weight, attrs = own[key]
        elif (int(mapping[u]), int(mapping[v])) in theirs:
            weight, attrs = theirs[(int(mapping[u]), int(mapping[v]))]
        else:
            weight, attrs = 1.0, np.ones(width)
        weights.append(weight)
        features.append(np.asarray(attrs, dtype=float).reshape(-1)[:width] if width else np.zeros(0))
    projected = copy.deepcopy(instance)
    projected.data = data
    projected.edge_weights = np.asarray(weights, dtype=np.float32)
    projected.edge_features = np.asarray(features, dtype=np.float32).reshape(len(weights), width)
    projected._nx_repr = None
    return projected


def structural_distance(instance, candidate, mapping) -> float:
    """Node and edge edits between the two graphs under the correspondence (the v2 measure's structure)."""
    return edit_cost(instance.data, candidate.data, mapping=mapping,
                     directed=instance.directed and candidate.directed)["structure"]


def nearest_valid_projection(instance, candidates, oracle, kind: str, input_label=None):
    """Project the candidates onto the input in order of structural distance under the correspondence and
    return (projection, candidate) for the first projection the oracle classifies differently from the
    input, or (None, None). Every check is one oracle call, counted as the generator's."""
    if input_label is None:
        input_label = oracle.predict(instance)
    ranked = []
    for candidate in candidates:
        mapping = correspondence(instance, candidate, kind)
        ranked.append((structural_distance(instance, candidate, mapping), len(ranked), candidate, mapping))
    ranked.sort(key=lambda item: (item[0], item[1]))
    for _, _, candidate, mapping in ranked:
        projected = project(instance, candidate, mapping)
        if np.array_equal(projected.data, instance.data):
            continue   # nothing of the candidate's structure lands on the input's nodes
        if oracle.predict(projected) != input_label:
            return projected, candidate
    return None, None
