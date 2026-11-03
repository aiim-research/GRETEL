#!/usr/bin/env python3
"""Unit checks for src/dataset/generators/treecycles_rand.py: deterministic given the seed, labels that
follow the cycle rule, the global numpy state untouched, and no duplicated instances when Dataset.read()
runs init() a second time. Needs the GRETEL environment (torch through the dataset base). Run from the
repo root with either of:

    python tests/treecycles_rand_unit.py
    python -m pytest tests/treecycles_rand_unit.py
"""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset.generators.treecycles_rand import TreeCyclesRand  # noqa: E402

CONTEXT = types.SimpleNamespace(logger=types.SimpleNamespace(info=lambda *a, **k: None))


def _generate(num_instances, seed, num_nodes=16, ratio=0.4):
    dataset = types.SimpleNamespace(instances=[])
    config = {'parameters': {'num_instances': num_instances, 'num_nodes_per_instance': num_nodes,
                             'ratio_nodes_in_cycles': ratio, 'seed': seed}}
    generator = TreeCyclesRand(CONTEXT, config, dataset)
    return generator, dataset


def _has_cycle(adj):
    import networkx as nx
    try:
        nx.find_cycle(nx.from_numpy_array(adj))
        return True
    except nx.NetworkXNoCycle:
        return False


def test_same_seed_same_graphs():
    _, a = _generate(20, seed=7)
    _, b = _generate(20, seed=7)
    assert len(a.instances) == len(b.instances) == 20
    for x, y in zip(a.instances, b.instances):
        assert x.label == y.label
        assert np.array_equal(x.data, y.data)


def test_different_seed_different_graphs():
    _, a = _generate(20, seed=1)
    _, b = _generate(20, seed=2)
    assert any(x.data.shape != y.data.shape or not np.array_equal(x.data, y.data)
               for x, y in zip(a.instances, b.instances))


def test_labels_match_the_oracle_rule():
    _, dataset = _generate(30, seed=3)
    for inst in dataset.instances:
        assert inst.data.shape == (16, 16)
        assert np.array_equal(inst.data, inst.data.T)
        assert _has_cycle(inst.data) == bool(inst.label)


def test_global_rng_untouched():
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    _generate(5, seed=0)
    assert np.array_equal(before, np.random.get_state()[1])


def test_generator_does_not_duplicate_on_second_init():
    generator, dataset = _generate(10, seed=5)
    assert len(dataset.instances) == 10
    first = [inst.data.copy() for inst in dataset.instances]
    generator.init()  # what Dataset.read() does after the constructor
    assert len(dataset.instances) == 10
    assert all(np.array_equal(a, inst.data) for a, inst in zip(first, dataset.instances))


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests:
        test()
        print('ok ', test.__name__)
    print(f'{len(tests)} passed')
