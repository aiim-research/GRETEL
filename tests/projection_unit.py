#!/usr/bin/env python3
"""Unit checks for src/explainer/future/utils/projection.py, the generator contract of protocol v2.

    python tests/projection_unit.py
"""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.explainer.future.utils.projection import (assignment_correspondence, correspondence,  # noqa: E402
                                                   nearest_valid_projection, project)


def graph(n, edges, features=None, gid=0):
    adj = np.zeros((n, n))
    for u, v in edges:
        adj[u, v] = adj[v, u] = 1
    nnz = int(np.count_nonzero(adj))
    return types.SimpleNamespace(id=gid, data=adj, directed=False, _nx_repr=None,
                                 node_features=np.zeros((n, 1)) if features is None else np.asarray(features, float),
                                 edge_weights=np.full(nnz, 2.0), edge_features=np.full((nnz, 1), 3.0))


class CycleOracle:
    """1 when the graph has a cycle (more edges than nodes minus components), as the Tree-Cycles rule."""
    calls = 0

    def predict(self, g):
        CycleOracle.calls += 1
        adj = g.data
        n = len(adj)
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for u, v in zip(*np.nonzero(np.triu(adj, 1))):
            ru, rv = find(u), find(v)
            if ru == rv:
                return 1
            parent[ru] = rv
        return 0


def test_projection_keeps_the_input_nodes_and_copies_structure():
    g = graph(3, [(0, 1)], features=[[1], [2], [3]])
    c = graph(5, [(0, 1), (1, 2), (0, 2), (3, 4)], features=[[9], [9], [9], [9], [9]])
    p = project(g, c, correspondence(g, c, 'index'))
    assert p.data.shape == (3, 3) and np.array_equal(p.node_features, g.node_features)
    assert {(u, v) for u, v in zip(*np.nonzero(np.triu(p.data, 1)))} == {(0, 1), (1, 2), (0, 2)}
    assert len(p.edge_weights) == np.count_nonzero(p.data) == p.edge_features.shape[0]


def test_edge_attributes_kept_or_taken_from_the_candidate():
    g = graph(3, [(0, 1)])
    g.edge_weights[:] = 7.0
    c = graph(3, [(0, 1), (1, 2)])
    p = project(g, c, correspondence(g, c, 'index'))
    weights = {tuple(map(int, e)): w for e, w in zip(zip(*np.nonzero(p.data)), p.edge_weights)}
    assert weights[(0, 1)] == 7.0 and weights[(1, 2)] == 2.0


def test_unmapped_input_nodes_keep_their_edges():
    g = graph(4, [(2, 3)])
    c = graph(2, [(0, 1)])
    p = project(g, c, correspondence(g, c, 'index'))
    assert p.data[0, 1] == 1 and p.data[2, 3] == 1


def test_assignment_matches_equal_features():
    g = graph(3, [], features=[[1, 0], [0, 1], [5, 5]])
    c = graph(3, [], features=[[5, 5], [1, 0], [0, 1]])
    assert assignment_correspondence(g, c).tolist() == [1, 2, 0]


def test_nearest_valid_projection_skips_invalid_candidates():
    tree = graph(4, [(0, 1), (1, 2), (2, 3)])
    far_cycle = graph(4, [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (1, 3)], gid=1)   # valid, distance 3
    near_tree = graph(4, [(0, 1), (1, 2)], gid=2)                                     # invalid, distance 1
    near_cycle = graph(4, [(0, 1), (1, 2), (2, 3), (0, 3)], gid=3)                   # valid, distance 1
    oracle = CycleOracle()
    projected, chosen = nearest_valid_projection(tree, [far_cycle, near_tree, near_cycle], oracle, 'index')
    assert chosen.id == 3 and oracle.predict(projected) == 1


def test_no_valid_projection():
    tree = graph(3, [(0, 1), (1, 2)])
    other_tree = graph(3, [(0, 2), (1, 2)])
    assert nearest_valid_projection(tree, [other_tree], CycleOracle(), 'index') == (None, None)


def test_a_bigger_candidate_is_cut_to_the_input():
    # The candidate's cycle sits on nodes the input does not have: the projection is no counterfactual.
    tree = graph(3, [(0, 1), (1, 2)])
    bigger = graph(6, [(0, 1), (1, 2), (3, 4), (4, 5), (3, 5)])
    assert nearest_valid_projection(tree, [bigger], CycleOracle(), 'index') == (None, None)


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests:
        test()
        print('ok ', test.__name__)
    print(f'{len(tests)} passed')
