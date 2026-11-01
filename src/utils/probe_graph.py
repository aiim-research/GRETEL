"""Lossless GraphInstance snapshots and validated recourse edits.

Reconstruction/Reversal can describe node changes. Recourse follows Alejandra's
proposal language: edge changes and node-feature values on existing nodes.
New node identities/bond attributes are not guessed by the application layer.
"""
import copy
import numpy as np

from src.utils.probe_common import validate_delta


def snapshot(instance):
    return {'id': instance.id, 'label': instance.label, 'directed': instance.directed,
            'data': instance.data.tolist(), 'data_shape': list(instance.data.shape),
            'node_features': instance.node_features.tolist(),
            'node_features_shape': list(instance.node_features.shape),
            'edge_features': instance.edge_features.tolist(), 'edge_weights': instance.edge_weights.tolist(),
            'edge_features_shape': list(instance.edge_features.shape),
            'graph_features': None if instance.graph_features is None else np.asarray(instance.graph_features).tolist()}


def restore(value, dataset=None):
    from src.dataset.instances.graph import GraphInstance
    data = np.asarray(value['data'], dtype=float)
    if 'data_shape' in value:
        data = data.reshape(tuple(value['data_shape']))
    if data.ndim != 2 or data.shape[0] != data.shape[1] or not np.isfinite(data).all():
        raise ValueError('Invalid saved adjacency matrix')
    if type(value['directed']) is not bool or (not value['directed'] and not np.array_equal(data, data.T)):
        raise ValueError('Saved adjacency does not match graph direction')
    attrs = {key: np.asarray(value[key], dtype=float) for key in ('node_features', 'edge_features', 'edge_weights')}
    for key in ('node_features', 'edge_features'):
        if key + '_shape' in value:
            attrs[key] = attrs[key].reshape(tuple(value[key + '_shape']))
    if any(not np.isfinite(array).all() for array in attrs.values()):
        raise ValueError('Nonfinite saved attributes')
    if attrs['node_features'].ndim != 2 or attrs['node_features'].shape[0] != len(data):
        raise ValueError('Invalid saved node features')
    nnz = np.count_nonzero(data)
    if (attrs['edge_features'].ndim != 2 or attrs['edge_features'].shape[0] != nnz
            or attrs['edge_weights'].shape != (nnz,)):
        raise ValueError('Invalid saved edge attributes')
    graph_features = value.get('graph_features')
    if graph_features is not None and not np.isfinite(np.asarray(graph_features, dtype=float)).all():
        raise ValueError('Nonfinite saved graph features')
    return GraphInstance(id=value['id'], label=value['label'], data=data, directed=value['directed'],
                         graph_features=None if graph_features is None else np.asarray(graph_features), dataset=dataset, **attrs)


def shown_nodes(instance):
    """Nodes before the trailing padding: isolated nodes with all-zero features."""
    data, features = instance.data, instance.node_features
    n = len(data)
    while n and not data[n - 1].any() and not data[:, n - 1].any() and not features[n - 1].any():
        n -= 1
    return n


def touches_hidden(original, counterfactual, shown):
    """Whether the counterfactual edits a node at or beyond `shown` (hidden padding)."""
    return bool(counterfactual.data[shown:].any() or counterfactual.data[:, shown:].any()
                or not np.array_equal(counterfactual.node_features[shown:], original.node_features[shown:]))


def delta_nodes(delta):
    nodes = set(delta.get('nodes_added', [])) | set(delta.get('nodes_removed', []))
    for key in ('edges_added', 'edges_removed'):
        nodes.update(node for edge in delta.get(key, []) for node in edge)
    nodes.update(change['node'] for change in delta.get('features_changed', []))
    return nodes


def graph_text(instance, label, feature_map=None, shown=None):
    """Prompt text of the graph; `shown` limits it to the leading nodes (hidden padding)."""
    data = instance.data
    shown = len(data) if shown is None else shown
    if data[shown:].any() or data[:, shown:].any():
        raise ValueError('Edges reach nodes beyond the shown ones')
    edges = [[int(u), int(v), float(data[u, v])] for u, v in zip(*np.nonzero(data))
             if instance.directed or u <= v]
    return (f'Nodes: {list(range(shown))}\nDirected: {instance.directed}\n'
            f'Edges [u,v,value]: {edges}\nFeature columns: {feature_map or {}}\n'
            f'Edge attribute order [u,v]: {[list(map(int, coord)) for coord in zip(*np.nonzero(data))]}\n'
            f'Edge weights: {instance.edge_weights.tolist()}\nEdge features: {instance.edge_features.tolist()}\n'
            f'Graph features: {None if instance.graph_features is None else np.asarray(instance.graph_features).tolist()}\n'
            f'Node features: {instance.node_features[:shown].tolist()}\nOracle predicted class: {label}\n')


def edge_attributes(instance):
    return {tuple(map(int, coord)): (instance.edge_weights[i], instance.edge_features[i])
            for i, coord in enumerate(zip(*np.nonzero(instance.data)))}


def unsupported_edge_changes(original, counterfactual):
    """TypedDelta does not describe value changes on edges present on both sides."""
    a, b = edge_attributes(original), edge_attributes(counterfactual)
    for edge in a.keys() & b.keys():
        if (original.data[edge] != counterfactual.data[edge] or a[edge][0] != b[edge][0]
                or not np.array_equal(a[edge][1], b[edge][1])):
            return True
    return False


def unsupported_changes(original, counterfactual):
    return (original.directed != counterfactual.directed
            or unsupported_edge_changes(original, counterfactual)
            or not np.array_equal(original.graph_features, counterfactual.graph_features))


def apply_edits(instance, delta, feature_map=None, edge_defaults=None):
    delta = validate_delta(delta, instance.directed, feature_match='transition')
    if delta['nodes_added'] or delta['nodes_removed']:
        raise NotImplementedError('Recourse does not invent attributes or mappings for added/removed nodes')
    if not any(delta.values()):
        raise ValueError('The proposed edit set is empty')
    candidate = copy.deepcopy(instance)
    data = candidate.data.copy()
    features = candidate.node_features.copy()
    n = len(data)
    for key, present in (('edges_added', False), ('edges_removed', True)):
        for u, v in delta[key]:
            if u >= n or v >= n:
                raise ValueError('Edge endpoint outside the factual graph')
            if bool(data[u, v]) != present:
                raise ValueError('Cannot add a present edge or remove an absent edge')
            data[u, v] = 0 if present else 1
            if not instance.directed:
                data[v, u] = data[u, v]
    feature_map = feature_map or {}
    for change in delta['features_changed']:
        node, feature = change['node'], change['feature']
        column = feature_map.get(feature, feature if type(feature) is int else None)
        if node >= n or type(column) is not int or not 0 <= column < features.shape[1]:
            raise ValueError('Unknown feature or node index')
        if features[node, column] != np.asarray(change['from'], dtype=features.dtype):
            raise ValueError('Feature from-value disagrees with the factual graph')
        features[node, column] = change['to']
        if not np.isfinite(features[node, column]) or features[node, column] == instance.node_features[node, column]:
            raise ValueError('Feature edit cannot be represented in the graph dtype')
    old = edge_attributes(instance)
    defaults = edge_defaults or {}
    width = instance.edge_features.shape[1]
    weights, edge_features = [], []
    for coord in zip(*np.nonzero(data)):
        if coord in old:
            weight, attributes = old[coord]
        else:
            if 'weight' not in defaults and not np.all(instance.edge_weights == 1):
                raise NotImplementedError('Adding weighted edges requires explicit edge_defaults.weight')
            if 'features' not in defaults and not np.all(instance.edge_features == 1):
                raise NotImplementedError('Adding attributed edges requires explicit edge_defaults.features')
            weight = defaults.get('weight', 1.0)
            attributes = np.asarray(defaults.get('features', [1.0] * width), dtype=float)
            if attributes.shape != (width,):
                raise ValueError('New edge feature vector has the wrong size')
        weights.append(weight)
        edge_features.append(attributes)
    candidate.data = data
    candidate.node_features = features
    candidate.edge_weights = np.asarray(weights, dtype=np.float32)
    candidate.edge_features = np.asarray(edge_features, dtype=np.float32).reshape((len(weights), width))
    if not np.isfinite(candidate.edge_weights).all() or not np.isfinite(candidate.edge_features).all():
        raise ValueError('Nonfinite edge attributes')
    candidate._nx_repr = None
    return candidate
