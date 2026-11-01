#!/usr/bin/env python3
"""Unit checks for src/utils/reconstruction_metrics.py.

Needs no third-party package. Run from the repo root with either of:

    python tests/reconstruction_metrics_unit.py
    python -m pytest tests/reconstruction_metrics_unit.py
"""
import importlib.util
import os

# Loaded from its file: importing the src.utils package pulls in torch
_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'utils', 'reconstruction_metrics.py')
_spec = importlib.util.spec_from_file_location('reconstruction_metrics', _path)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)


def test_empty_set_conventions():
    assert _m.calculate_metrics(set(), set())['f1'] == 1.0
    assert _m.calculate_metrics(set(), {1})['f1'] == 0.0
    assert _m.calculate_metrics({1}, set())['f1'] == 0.0


def test_partial_overlap():
    scores = _m.calculate_metrics({1, 2, 3, 4}, {3, 4, 5})
    assert scores['precision'] == 2 / 3
    assert scores['recall'] == 0.5
    assert abs(scores['f1'] - 4 / 7) < 1e-12
    assert scores['jaccard'] == 0.4


def test_undirected_edges_match_regardless_of_order():
    truth = {'edges_removed': [[1, 2]], 'edges_added': [], 'features_changed': []}
    pred = {'edges_removed': [[2, 1]], 'edges_added': [], 'features_changed': []}
    assert _m.score_delta(truth, pred)['edges_removed']['f1'] == 1.0
    assert _m.score_delta(truth, pred, directed=True)['edges_removed']['f1'] == 0.0


def test_added_and_removed_do_not_match_each_other():
    truth = {'edges_removed': [[1, 2]], 'edges_added': [], 'features_changed': []}
    pred = {'edges_removed': [], 'edges_added': [[1, 2]], 'features_changed': []}
    scores = _m.score_delta(truth, pred)
    assert scores['edges']['f1'] == 0.0
    assert scores['edges_removed']['f1'] == 0.0
    assert scores['edges_added']['f1'] == 0.0


def test_feature_changes_match_on_node_and_name_only():
    truth = {'edges_removed': [], 'edges_added': [],
             'features_changed': [{'node': 3, 'feature': 'charge', 'from': 1.0, 'to': 0.0}]}
    pred = {'edges_removed': [], 'edges_added': [],
            'features_changed': [{'node': 3, 'feature': 'charge'}]}
    assert _m.score_delta(truth, pred)['features_changed']['f1'] == 1.0
    pred['features_changed'][0]['node'] = 4
    assert _m.score_delta(truth, pred)['features_changed']['f1'] == 0.0


if __name__ == '__main__':
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith('test_')]
    for test in tests:
        test()
        print(f'ok  {test.__name__}')
    print(f'{len(tests)} passed')
