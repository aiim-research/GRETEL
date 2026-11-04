#!/usr/bin/env python3
"""Unit checks for src/utils/metrics/correspondence.py, the edit cost under a node correspondence.

Known cases, the properties its docstring claims (zero, symmetry under the inverse correspondence,
monotonicity, triangle inequality under composition) on random graphs, and the regression against the
cell-wise GED on same-size pairs. numpy only. Run from the repository root:

    python tests/edit_distance_unit.py
    python -m pytest tests/edit_distance_unit.py
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.metrics.correspondence import (compose_correspondences, edit_cost,  # noqa: E402
                                              identity_correspondence, inverse_correspondence)
from src.utils.metrics.ged import graph_edit_distance_metric  # noqa: E402


def graph(n, edges):
    adj = np.zeros((n, n))
    for u, v in edges:
        adj[u, v] = adj[v, u] = 1
    return adj


def random_graph(rng, n, p=0.3, directed=False):
    adj = (rng.random((n, n)) < p).astype(float)
    if not directed:
        adj = np.triu(adj, 1)
        adj = adj + adj.T
    return adj


def random_correspondence(rng, n1, n2):
    k = int(rng.integers(0, min(n1, n2) + 1))
    mapping = np.full(n1, -1)
    mapping[rng.choice(n1, k, replace=False)] = rng.choice(n2, k, replace=False)
    return mapping


def test_identical_graphs_cost_nothing():
    g = graph(4, [(0, 1), (1, 2), (2, 3)])
    x = np.arange(8.0).reshape(4, 2)
    d = edit_cost(g, g, x, x)
    assert d['total'] == 0 and d['structure'] == 0


def test_node_insertion_with_an_edge():
    # The padded cell-wise GED charged 1 for this (the new edge, no node); here: one node and one edge.
    g, h = graph(3, [(0, 1), (1, 2)]), graph(4, [(0, 1), (1, 2), (2, 3)])
    d = edit_cost(g, h)
    assert (d['nodes_inserted'], d['edges_inserted'], d['nodes_deleted'], d['edges_deleted']) == (1, 1, 0, 0)
    assert edit_cost(h, g)['nodes_deleted'] == 1 and edit_cost(h, g)['edges_deleted'] == 1
    assert edit_cost(g, h, node_cost=0.5)['total'] == 1.5


def test_edges_between_common_and_extra_nodes_are_charged():
    # The cell-wise count skips the side strips: an extra node joined to every old one cost 0 there.
    g, h = graph(3, []), graph(4, [(0, 3), (1, 3), (2, 3)])
    assert edit_cost(g, h)['edges_inserted'] == 3


def test_self_loop_is_one_edit():
    g = np.zeros((2, 2))
    h = g.copy()
    h[0, 0] = 1
    assert edit_cost(g, h)['edges_inserted'] == 1


def test_features_only_on_mapped_nodes():
    g, h = graph(2, [(0, 1)]), graph(3, [(0, 1)])
    x, y = np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[3.0, 4.0], [1.0, 1.0], [9.0, 9.0]])
    d = edit_cost(g, h, x, y)
    assert d['feature_distance'] == 5.0 and d['nodes_inserted'] == 1


def test_a_correspondence_can_make_a_relabelling_free():
    g = graph(3, [(0, 1)])
    h = graph(3, [(1, 2)])
    assert edit_cost(g, h)['structure'] == 2
    assert edit_cost(g, h, mapping=[1, 2, 0])['structure'] == 0


def test_invalid_correspondences_are_rejected():
    g = graph(3, [])
    for bad in ([0, 0, 1], [0, 1], [0, 1, 3], [0, -2, 1]):
        try:
            edit_cost(g, g, mapping=bad)
        except ValueError:
            continue
        raise AssertionError(f'accepted {bad}')


def test_same_size_identity_matches_the_cellwise_ged():
    rng = np.random.default_rng(0)
    for directed in (False, True):
        for _ in range(200):
            n = int(rng.integers(1, 12))
            a, b = random_graph(rng, n, directed=directed), random_graph(rng, n, directed=directed)
            np.fill_diagonal(a, 0)
            np.fill_diagonal(b, 0)
            assert edit_cost(a, b, directed=directed)['total'] == graph_edit_distance_metric(a, b, directed)


def test_zero_only_without_changes():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n1, n2 = rng.integers(1, 7, 2)
        a, b = random_graph(rng, n1), random_graph(rng, n2)
        x, y = rng.integers(0, 2, (n1, 2)).astype(float), rng.integers(0, 2, (n2, 2)).astype(float)
        m = random_correspondence(rng, n1, n2)
        d = edit_cost(a, b, x, y, mapping=m)['total']
        bijective = n1 == n2 and (m >= 0).all()
        same = bijective and np.array_equal(a, b[np.ix_(m, m)]) and np.array_equal(x, y[m])
        assert (d == 0) == same


def test_symmetry_under_the_inverse():
    rng = np.random.default_rng(2)
    for _ in range(300):
        n1, n2 = rng.integers(1, 8, 2)
        a, b = random_graph(rng, n1), random_graph(rng, n2)
        x, y = rng.random((n1, 3)), rng.random((n2, 3))
        m = random_correspondence(rng, n1, n2)
        forward = edit_cost(a, b, x, y, mapping=m)
        backward = edit_cost(b, a, y, x, mapping=inverse_correspondence(m, n2))
        assert np.isclose(forward['total'], backward['total'])
        assert (forward['nodes_deleted'], forward['edges_deleted']) == (backward['nodes_inserted'], backward['edges_inserted'])


def test_one_more_edit_never_lowers_the_cost():
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = int(rng.integers(2, 9))
        a, b = random_graph(rng, n), random_graph(rng, n)
        before = edit_cost(a, b)['total']
        # Toggle one cell where a and b agree: one more difference.
        agree = [(u, v) for u, v in itertools.combinations(range(n), 2) if a[u, v] == b[u, v]]
        if not agree:
            continue
        u, v = agree[int(rng.integers(len(agree)))]
        c = b.copy()
        c[u, v] = c[v, u] = 1 - c[u, v]
        assert edit_cost(a, c)['total'] == before + 1


def test_triangle_inequality_under_composition():
    rng = np.random.default_rng(4)
    for _ in range(300):
        n1, n2, n3 = rng.integers(1, 7, 3)
        a, b, c = random_graph(rng, n1), random_graph(rng, n2), random_graph(rng, n3)
        x, y, z = rng.random((n1, 2)), rng.random((n2, 2)), rng.random((n3, 2))
        m1, m2 = random_correspondence(rng, n1, n2), random_correspondence(rng, n2, n3)
        direct = edit_cost(a, c, x, z, mapping=compose_correspondences(m1, m2))['total']
        via = edit_cost(a, b, x, y, mapping=m1)['total'] + edit_cost(b, c, y, z, mapping=m2)['total']
        assert direct <= via + 1e-9


def test_identity_correspondence():
    assert identity_correspondence(3, 2).tolist() == [0, 1, -1]
    assert identity_correspondence(2, 3).tolist() == [0, 1]


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests:
        test()
        print('ok ', test.__name__)
    print(f'{len(tests)} passed')
