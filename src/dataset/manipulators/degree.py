import numpy as np

from src.dataset.manipulators.base import BaseManipulator


class NodeDegree(BaseManipulator):
    """One-hot node degree (degrees above ``max_degree``, default 64, share the last slot) as the only
    structural feature, for datasets without node features (IMDB-BINARY). On IMDB-BINARY a GCN learns
    nothing from the degree as a single number (50 %) and 72 % (test) from its one-hot encoding.
    ``NodeCentrality`` computes betweenness, closeness and a Laplacian spectrum, and the minimizers
    recompute the features of every candidate graph, which on graphs of hundreds of nodes would cost
    seconds per oracle call."""

    def node_info(self, instance):
        max_degree = self.local_config.get('parameters', {}).get('max_degree', 64)
        degrees = np.minimum((np.asarray(instance.data) != 0).sum(axis=1), max_degree)
        onehot = np.zeros((len(degrees), max_degree + 1), dtype=np.float32)
        onehot[np.arange(len(degrees)), degrees] = 1.0
        return {f"degree_{d}": list(onehot[:, d]) for d in range(max_degree + 1)}
