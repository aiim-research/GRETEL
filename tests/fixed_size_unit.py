#!/usr/bin/env python3
"""check_fixed_size (src/core/explainer_base.py): the explainers that need one graph size fail clearly on a
dataset of several sizes (protocol v2, no padding) and run as before on padded or fixed-size datasets.

    python tests/fixed_size_unit.py
"""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.explainer_base import check_fixed_size  # noqa: E402


def dataset(*sizes):
    return types.SimpleNamespace(instances=[types.SimpleNamespace(data=np.zeros((n, n))) for n in sizes])


def component(flag):
    return type('Component', (), {'requires_fixed_size': flag})()


def test_fixed_size_component_on_several_sizes_fails():
    try:
        check_fixed_size(component(True), dataset(3, 5))
    except ValueError as exc:
        assert 'one size' in str(exc)
    else:
        raise AssertionError('no error')


def test_fixed_size_component_on_one_size_runs():
    check_fixed_size(component(True), dataset(7, 7, 7))   # padded or fixed-size datasets: unchanged


def test_other_components_never_fail():
    check_fixed_size(component(False), dataset(3, 5))
    check_fixed_size(object(), dataset(3, 5))
    check_fixed_size(component(True), None)


def test_the_flags():
    from src.explainer.future.meta.minimizer.dbs import DBS
    from src.explainer.future.search.dfs import DFS
    from src.explainer.future.search.ofs import OFS
    from src.explainer.future.metaheuristic.local_search.local_search import LocalSearch
    assert DBS.requires_fixed_size and DFS.requires_fixed_size
    assert not OFS.requires_fixed_size and not LocalSearch.requires_fixed_size


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests:
        test()
        print('ok ', test.__name__)
    print(f'{len(tests)} passed')
