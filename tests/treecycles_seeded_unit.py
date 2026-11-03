#!/usr/bin/env python3
"""Unit checks for src/dataset/generators/treecycles_seeded.py.

The generation function needs numpy and networkx; the generator class also
needs the GRETEL environment (torch through the dataset base). Run from the
repo root with either of:

    python tests/treecycles_seeded_unit.py
    python -m pytest tests/treecycles_seeded_unit.py
"""
import importlib.util
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.dataset.generators.treecycles_seeded import TreeCyclesSeeded, generate_tree_cycles
    HAVE_CLASS = True
except ImportError:
    # Without torch only the pure function can be loaded: pull it off the
    # file with the class's imports stubbed out
    HAVE_CLASS = False
    for name in ('src', 'src.dataset', 'src.dataset.generators', 'src.dataset.generators.treecycles_rand',
                 'src.dataset.instances', 'src.dataset.instances.graph'):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules['src.dataset.generators.treecycles_rand'].TreeCyclesRand = object
    sys.modules['src.dataset.instances.graph'].GraphInstance = object
    _path = os.path.join(os.path.dirname(__file__), '..', 'src', 'dataset', 'generators', 'treecycles_seeded.py')
    _spec = importlib.util.spec_from_file_location('treecycles_seeded', _path)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    generate_tree_cycles = _module.generate_tree_cycles


def _has_cycle(adj):
    import networkx as nx
    try:
        nx.find_cycle(nx.from_numpy_array(adj))
        return True
    except nx.NetworkXNoCycle:
        return False


def test_same_seed_same_graphs():
    a = generate_tree_cycles(20, 16, 0.4, seed=7)
    b = generate_tree_cycles(20, 16, 0.4, seed=7)
    assert len(a) == len(b) == 20
    for (adj_a, lbl_a), (adj_b, lbl_b) in zip(a, b):
        assert lbl_a == lbl_b
        assert np.array_equal(adj_a, adj_b)


def test_different_seed_different_graphs():
    a = generate_tree_cycles(20, 16, 0.4, seed=1)
    b = generate_tree_cycles(20, 16, 0.4, seed=2)
    assert any(not np.array_equal(x[0], y[0]) for x, y in zip(a, b))


def test_labels_match_the_oracle_rule():
    for adj, label in generate_tree_cycles(30, 16, 0.4, seed=3):
        assert adj.shape == (16, 16)
        assert np.array_equal(adj, adj.T)
        assert _has_cycle(adj) == bool(label)


def test_global_rng_untouched():
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    generate_tree_cycles(5, 16, 0.4, seed=0)
    assert np.array_equal(before, np.random.get_state()[1])


def test_generator_does_not_duplicate_on_second_init():
    if not HAVE_CLASS:
        print('skip test_generator_does_not_duplicate_on_second_init (needs torch)')
        return
    context = types.SimpleNamespace(logger=types.SimpleNamespace(info=lambda *a, **k: None))
    dataset = types.SimpleNamespace(instances=[])
    config = {'parameters': {'num_instances': 10, 'num_nodes_per_instance': 16,
                             'ratio_nodes_in_cycles': 0.4, 'seed': 5}}
    generator = TreeCyclesSeeded(context, config, dataset)
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
