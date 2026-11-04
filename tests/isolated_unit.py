#!/usr/bin/env python3
"""The isolated-node policy of protocol v2 (src/utils/isolated.py, the DeleteIsolatedNodes manipulator and
the oracle hook in src/core/oracle_base.py).

    python tests/isolated_unit.py
"""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.isolated import drop_isolated, isolated_nodes, materialize  # noqa: E402


def graph(n, edges, features=None):
    adj = np.zeros((n, n))
    for u, v in edges:
        adj[u, v] = adj[v, u] = 1
    nnz = int(np.count_nonzero(adj))
    return types.SimpleNamespace(data=adj, node_features=np.arange(n, dtype=float).reshape(n, 1) if features is None else features,
                                 edge_weights=np.arange(nnz, dtype=float), edge_features=np.ones((nnz, 1)), _nx_repr=None)


def test_isolated_nodes():
    assert isolated_nodes(graph(4, [(0, 2)]).data).tolist() == [1, 3]
    selfloop = graph(2, [])
    selfloop.data[1, 1] = 1
    assert isolated_nodes(selfloop.data).tolist() == [0]


def test_drop_keeps_edges_and_their_attributes_in_order():
    g = graph(5, [(0, 2), (2, 4)])
    out, keep = drop_isolated(g)
    assert keep.tolist() == [0, 2, 4] and out.data.shape == (3, 3)
    assert out.node_features.ravel().tolist() == [0.0, 2.0, 4.0]
    assert np.count_nonzero(out.data) == len(out.edge_weights) == 4


def test_nothing_to_drop_or_everything_isolated():
    g = graph(3, [(0, 1), (1, 2)])
    assert drop_isolated(g)[0] is g
    empty = graph(3, [])
    assert drop_isolated(empty)[0] is empty


def test_materialize_records_deleted_nodes():
    out, mapping = materialize(graph(4, [(1, 3)]), 'delete')
    assert mapping.tolist() == [-1, 0, -1, 1] and out.data.shape == (2, 2)
    out, mapping = materialize(graph(4, [(1, 3)]), 'keep')
    assert mapping.tolist() == [0, 1, 2, 3] and out.data.shape == (4, 4)


def test_oracle_sees_the_graph_without_isolated_nodes():
    from src.core.oracle_base import Oracle

    class SizeOracle(Oracle):
        def init(self):
            pass

        def real_fit(self):
            pass

        def _real_predict(self, g):
            return len(g.data)

        def _real_predict_proba(self, g):
            return [len(g.data)]

    oracle = SizeOracle.__new__(SizeOracle)
    oracle._call_counter = 0
    g = graph(4, [(0, 1)])
    oracle.dataset = types.SimpleNamespace(isolated_nodes='keep')
    assert oracle.predict(g) == 4
    oracle.dataset = types.SimpleNamespace(isolated_nodes='delete')
    assert oracle.predict(g) == 2 and oracle.get_calls_count() == 2


def test_manipulator_strips_at_load_and_declares_the_policy():
    from src.dataset.manipulators.isolated import DeleteIsolatedNodes
    instances = [graph(3, [(0, 1)]), graph(2, [(0, 1)])]
    dataset = types.SimpleNamespace(instances=instances, _num_nodes=5, _num_nodes_values=[3, 2])
    manipulator = DeleteIsolatedNodes.__new__(DeleteIsolatedNodes)
    manipulator.dataset = dataset
    manipulator.process()
    assert dataset.isolated_nodes == 'delete' and [len(i.data) for i in instances] == [2, 2]
    assert dataset._num_nodes_values is None


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests:
        test()
        print('ok ', test.__name__)
    print(f'{len(tests)} passed')
