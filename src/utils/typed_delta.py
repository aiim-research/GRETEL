"""Typed edit set between a graph G and a counterfactual G'.

Nodes are matched by index, the same assumption graph_edit_distance_metric
makes: node i of G is node i of G'. When the two graphs differ in size, the
extra indices are reported as added or removed nodes, and every edge that
touches them lands in edges_added or edges_removed.

The delta has one list per edit type:

    nodes_added, nodes_removed     node indices
    edges_added, edges_removed     [u, v] pairs, u < v for undirected graphs
    features_changed               {node, feature, from, to} for nodes present
                                   in both graphs

Edge features and edge weights are outside this schema. The paper generation
stage rejects changes to these attributes on existing edges rather than
treating them as an empty delta. Undirected self-loops are included.
"""
import numpy as np


def _edge_set(adj, directed):
    rows, cols = np.nonzero(adj)
    if directed:
        return {(int(i), int(j)) for i, j in zip(rows, cols)}
    return {(int(min(i, j)), int(max(i, j))) for i, j in zip(rows, cols)}


def typed_delta(adj, adj_cf, node_features, node_features_cf, directed=False,
                feature_columns=None, feature_names=None, atol=0.0):
    """Compute the typed edit set between two graphs given as arrays.

    Args:
    - adj, adj_cf: adjacency matrices of G and G'
    - node_features, node_features_cf: node feature matrices (nodes x features)
    - directed: whether edges are ordered pairs
    - feature_columns: feature columns to compare (default: all). Use it to
      leave out features a dataset manipulator derives from the topology
      (degree, centralities), which change as a side effect of edge edits.
    - feature_names: {column index: name} used to label features_changed
    - atol: absolute tolerance below which a feature value counts as unchanged
    """
    n, n_cf = adj.shape[0], adj_cf.shape[0]

    edges = _edge_set(adj, directed)
    edges_cf = _edge_set(adj_cf, directed)

    common = min(n, n_cf)
    features_changed = []
    if node_features is not None and node_features_cf is not None:
        num_columns = min(node_features.shape[1], node_features_cf.shape[1])
        columns = range(num_columns) if feature_columns is None else feature_columns
        for node in range(common):
            for col in columns:
                before = float(node_features[node, col])
                after = float(node_features_cf[node, col])
                if abs(before - after) > atol:
                    name = feature_names.get(col, col) if feature_names else col
                    features_changed.append(
                        {"node": node, "feature": name, "from": before, "to": after})

    delta = {
        "nodes_added": list(range(n, n_cf)),
        "nodes_removed": list(range(n_cf, n)),
        "edges_added": [list(e) for e in sorted(edges_cf - edges)],
        "edges_removed": [list(e) for e in sorted(edges - edges_cf)],
        "features_changed": features_changed,
    }
    delta["size"] = {key: len(value) for key, value in delta.items()}
    return delta


def typed_delta_from_instances(instance, cf_instance, feature_columns=None,
                               feature_names=None, atol=0.0):
    """typed_delta for two GraphInstance objects."""
    return typed_delta(instance.data, cf_instance.data,
                       instance.node_features, cf_instance.node_features,
                       directed=instance.directed and cf_instance.directed,
                       feature_columns=feature_columns,
                       feature_names=feature_names,
                       atol=atol)
