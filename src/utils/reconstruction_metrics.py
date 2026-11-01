"""Scoring of a reconstructed edit set against the typed-delta ground truth.

Pure functions, no framework imports, so they can be unit-tested and reused
outside a pipeline run.

Conventions (frozen, they match the thesis protocol):
- both sets empty            -> 1.0 on every metric (correct abstention)
- truth empty, prediction not -> 0.0 (the judge invented an edit)
- truth not empty, prediction empty -> 0.0 (nothing recovered)

Edits are matched strictly:
- edges as canonical pairs (min, max) for undirected graphs
- feature_match=identity: (node, feature), the thesis's legacy convention
- feature_match=transition: (node, feature, from, to), exact numeric values
"""

EDIT_TYPES = ('edges_added', 'edges_removed', 'features_changed')


def calculate_metrics(set_true, set_pred):
    """Jaccard, precision, recall and F1 between two sets."""
    if not set_true and not set_pred:
        return {'jaccard': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0}

    tp = len(set_true & set_pred)
    union = len(set_true | set_pred)
    precision = tp / len(set_pred) if set_pred else 0.0
    recall = tp / len(set_true) if set_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {'jaccard': tp / union if union else 0.0,
            'precision': precision, 'recall': recall, 'f1': f1}


def edge_key(edge, directed=False):
    u, v = int(edge[0]), int(edge[1])
    return (u, v) if directed else (min(u, v), max(u, v))


def feature_key(change, feature_match='identity'):
    key = (int(change['node']), str(change['feature']))
    return key + (change['from'], change['to']) if feature_match == 'transition' else key


def delta_to_sets(delta, directed=False, feature_match='identity'):
    """Turn a typed delta (ground truth or extraction) into one set per type."""
    sets = {}
    for edit_type in ('edges_added', 'edges_removed'):
        sets[edit_type] = {edge_key(e, directed) for e in delta.get(edit_type) or []}
    sets['features_changed'] = {feature_key(c, feature_match) for c in delta.get('features_changed') or []}
    for key in ('nodes_added', 'nodes_removed'):
        sets[key] = set(delta.get(key) or [])
    return sets


def score_delta(truth, prediction, directed=False, feature_match='identity'):
    """Per-type metrics plus a structural aggregate over both edge types."""
    if feature_match not in ('identity', 'transition'):
        raise ValueError('feature_match must be identity or transition')
    truth_sets = delta_to_sets(truth, directed, feature_match)
    pred_sets = delta_to_sets(prediction, directed, feature_match)

    scores = {edit_type: calculate_metrics(truth_sets[edit_type], pred_sets[edit_type])
              for edit_type in EDIT_TYPES}

    # Edges tagged by direction, so an added edge does not match a removed one
    truth_edges = {('+',) + e for e in truth_sets['edges_added']} | {('-',) + e for e in truth_sets['edges_removed']}
    pred_edges = {('+',) + e for e in pred_sets['edges_added']} | {('-',) + e for e in pred_sets['edges_removed']}
    scores['edges'] = calculate_metrics(truth_edges, pred_edges)
    for key in ('nodes_added', 'nodes_removed'):
        scores[key] = calculate_metrics(truth_sets[key], pred_sets[key])
    return scores
