import numpy as np

from src.dataset.manipulators.base import BaseManipulator


class NodeTriangles(BaseManipulator):
    """The number of triangles each node belongs to, and its degree: the features a GCN oracle needs on
    the triangle-threshold datasets (with the degree alone, message passing cannot count triangles).
    Cheap enough to recompute on every candidate graph (one dense matrix product)."""

    def node_info(self, instance):
        a = (np.asarray(instance.data) != 0).astype(np.float64)
        np.fill_diagonal(a, 0)
        a = np.maximum(a, a.T)
        return {"triangles": list((a @ a * a).sum(axis=1) / 2), "degree": list(a.sum(axis=1))}
