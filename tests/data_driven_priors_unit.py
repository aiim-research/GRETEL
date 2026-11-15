#!/usr/bin/env python3
"""The type-pair prior of DFS and DBS (src/explainer/future/utils/data_driven_priors.py, protocol v2).

* with the node index as the type, no normalisation and fold -1 it is exactly Abrate and Bonchi's index prior;
* it runs on graphs of several sizes, counts only the training graphs of the fold, and a node type unseen in
  training weighs nothing;
* with one-hot types it gives an edge the frequency of its pair of types in each class.

    python tests/data_driven_priors_unit.py
"""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.explainer.future.utils import data_driven_priors as ddp  # noqa: E402


def graph(gid, label, adj, x=None):
    adj = np.asarray(adj, dtype=float)
    return types.SimpleNamespace(id=gid, label=label, data=adj, node_features=None if x is None else np.asarray(x, float),
                                 is_directed=False)


def random_graph(rng, gid, label, n, x=None):
    a = np.triu((rng.random((n, n)) < 0.35).astype(float), 1)
    return graph(gid, label, a + a.T, x)


def dataset(instances, splits=None):
    ds = types.SimpleNamespace(instances=instances, splits=splits or [])

    def get_split_indices(fold_id=-1):
        if fold_id == -1:
            return {'train': list(range(len(instances))), 'test': list(range(len(instances)))}
        return ds.splits[fold_id]
    ds.get_split_indices = get_split_indices
    return ds


def test_index_types_unnormalised_equal_the_index_prior():
    rng = np.random.default_rng(0)
    ds = dataset([random_graph(rng, k, k % 2, 7) for k in range(30)])
    ddp.clear_cache()
    for inst in ds.instances[:6]:
        for f_E in (0, 1):
            old = ddp.prior_weights(ds, inst, f_E, {'prior': 'index'})
            new = ddp.prior_weights(ds, inst, f_E, {'prior': 'type_pair', 'node_type': {'kind': 'index'}, 'normalize': False})
            off = ~np.eye(7, dtype=bool)
            assert np.array_equal(old[off], new[off]), (inst.id, f_E)


def test_several_sizes_and_training_graphs_only():
    rng = np.random.default_rng(1)
    onehot = lambda n: np.eye(3)[rng.integers(0, 3, n)]
    graphs = [random_graph(rng, k, k % 2, n, np.hstack([rng.random((n, 2)), onehot(n)])) for k, n in enumerate([5, 9, 6, 12, 7, 8])]
    ds = dataset(graphs, splits=[{'train': [0, 1, 2, 3], 'test': [4, 5]}])
    ddp.clear_cache()
    spec = {'prior': 'type_pair', 'node_type': {'kind': 'columns', 'columns': [2, 5]}, 'normalize': True}
    w = ddp.prior_weights(ds, graphs[5], 0, spec, fold_id=0)
    assert w.shape == (8, 8)
    prior = ddp.type_pair_prior(ds, 0, spec['node_type'], True)
    assert prior.train_ids == {0, 1, 2, 3} and sum(prior.graphs.values()) == 4
    # a test graph changes nothing in the counts
    graphs[4].data[:] = 1 - graphs[4].data
    ddp.clear_cache()
    assert np.array_equal(w, ddp.prior_weights(ds, graphs[5], 0, spec, fold_id=0))


def test_one_hot_pair_frequencies():
    # class 0: one A-B edge per graph; class 1: one A-A edge per graph (types: column 0 = A, column 1 = B)
    A, B = [1, 0], [0, 1]
    g0 = [graph(k, 0, [[0, 1, 0], [1, 0, 0], [0, 0, 0]], [A, B, A]) for k in range(2)]
    g1 = [graph(10 + k, 1, [[0, 1, 0], [1, 0, 0], [0, 0, 0]], [A, A, B]) for k in range(3)]
    ds = dataset(g0 + g1, splits=[{'train': list(range(5)), 'test': []}])
    ddp.clear_cache()
    spec = {'prior': 'type_pair', 'node_type': {'kind': 'columns', 'columns': [0, 2]}, 'normalize': True}
    x = graph(99, 0, [[0, 1, 0], [1, 0, 1], [0, 1, 0]], [A, B, A])    # edges A-B (0,1) and B-A (1,2), absent A-A (0,2)
    w = ddp.prior_weights(ds, x, 0, spec, fold_id=0)
    assert np.isclose(w[0, 1], 1.0) and np.isclose(w[1, 2], 1.0)        # A-B: 1 per graph in class 0, 0 in class 1
    assert np.isclose(w[0, 2], 1.0)                                     # absent A-A: D⁻ - D⁺ = 1 - 0
    # an unseen type weighs nothing
    y = graph(98, 0, [[0, 1], [1, 0]], [[1, 1], A])
    wy = ddp.prior_weights(ds, y, 0, spec, fold_id=0)
    assert wy[0, 1] == 0.0 and wy[1, 0] == 0.0                         # (the diagonal is never a candidate)


def test_kmeans_types():
    rng = np.random.default_rng(2)
    graphs = [random_graph(rng, k, k % 2, n, rng.normal(size=(n, 4))) for k, n in enumerate([6, 10, 8, 7, 9, 11])]
    ds = dataset(graphs)
    ddp.clear_cache()
    w = ddp.prior_weights(ds, graphs[0], 1, {'prior': 'type_pair', 'node_type': {'kind': 'kmeans', 'k': 3, 'seed': 0}})
    assert w.shape == (6, 6) and np.all(np.isfinite(w))


def test_the_flags():
    from src.explainer.future.meta.minimizer.dbs import DBS
    from src.explainer.future.search.dfs import DFS
    for cls in (DBS, DFS):
        obj = object.__new__(cls)
        obj.local_config = {'parameters': {}}
        assert obj.requires_fixed_size
        obj.local_config = {'parameters': {'prior': 'type_pair'}}
        assert not obj.requires_fixed_size


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests:
        test()
        print('ok ', test.__name__)
    print(f'{len(tests)} passed')
