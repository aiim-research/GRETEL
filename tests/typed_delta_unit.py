#!/usr/bin/env python3
"""Unit checks for src/utils/typed_delta.py.

Needs only numpy. Run from the repo root with either of:

    python tests/typed_delta_unit.py
    python -m pytest tests/typed_delta_unit.py
"""
import importlib.util
import os

import numpy as np

# Loaded from its file: importing the src.utils package pulls in torch
_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'utils', 'typed_delta.py')
_spec = importlib.util.spec_from_file_location('typed_delta', _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
typed_delta = _module.typed_delta


def _adj(n, edges, directed=False):
    adj = np.zeros((n, n))
    for i, j in edges:
        adj[i, j] = 1
        if not directed:
            adj[j, i] = 1
    return adj


def test_identical_graphs_give_empty_delta():
    adj = _adj(4, [(0, 1), (1, 2)])
    x = np.ones((4, 2))
    delta = typed_delta(adj, adj, x, x)
    assert all(size == 0 for size in delta['size'].values())


def test_undirected_edges_are_canonical_pairs():
    adj = _adj(4, [(0, 1), (1, 2)])
    adj_cf = _adj(4, [(1, 0), (2, 3)])
    x = np.zeros((4, 1))
    delta = typed_delta(adj, adj_cf, x, x)
    assert delta['edges_added'] == [[2, 3]]
    assert delta['edges_removed'] == [[1, 2]]


def test_directed_edges_keep_orientation():
    adj = _adj(3, [(0, 1)], directed=True)
    adj_cf = _adj(3, [(1, 0)], directed=True)
    x = np.zeros((3, 1))
    delta = typed_delta(adj, adj_cf, x, x, directed=True)
    assert delta['edges_added'] == [[1, 0]]
    assert delta['edges_removed'] == [[0, 1]]


def test_feature_changes_are_named_and_filtered():
    adj = _adj(2, [(0, 1)])
    x = np.array([[1.0, 5.0], [0.0, 3.0]])
    x_cf = np.array([[0.0, 6.0], [0.0, 3.0]])
    names = {0: 'charge', 1: 'degree'}

    delta = typed_delta(adj, adj, x, x_cf, feature_names=names)
    assert delta['features_changed'] == [
        {'node': 0, 'feature': 'charge', 'from': 1.0, 'to': 0.0},
        {'node': 0, 'feature': 'degree', 'from': 5.0, 'to': 6.0},
    ]

    # Only the intrinsic feature: the derived degree is left out
    delta = typed_delta(adj, adj, x, x_cf, feature_columns=[0], feature_names=names)
    assert [c['feature'] for c in delta['features_changed']] == ['charge']


def test_tolerance_ignores_small_feature_changes():
    adj = _adj(2, [(0, 1)])
    x = np.array([[1.0], [0.0]])
    x_cf = np.array([[1.0 + 1e-7], [0.5]])
    delta = typed_delta(adj, adj, x, x_cf, atol=1e-6)
    assert delta['features_changed'] == [{'node': 1, 'feature': 0, 'from': 0.0, 'to': 0.5}]


def test_size_mismatch_reports_nodes_and_their_edges():
    adj = _adj(3, [(0, 1)])
    adj_cf = _adj(4, [(0, 1), (2, 3)])
    delta = typed_delta(adj, adj_cf, np.zeros((3, 1)), np.zeros((4, 1)))
    assert delta['nodes_added'] == [3]
    assert delta['nodes_removed'] == []
    assert delta['edges_added'] == [[2, 3]]

    delta = typed_delta(adj_cf, adj, np.zeros((4, 1)), np.zeros((3, 1)))
    assert delta['nodes_removed'] == [3]
    assert delta['edges_removed'] == [[2, 3]]


if __name__ == '__main__':
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith('test_')]
    for test in tests:
        test()
        print(f'ok  {test.__name__}')
    print(f'{len(tests)} passed')
