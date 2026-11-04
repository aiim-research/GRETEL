import numpy as np

from src.dataset.manipulators.base import BaseManipulator
from src.utils.isolated import drop_isolated


class DeleteIsolatedNodes(BaseManipulator):
    """Protocol v2 isolated-node policy (src/utils/isolated.py): removes the nodes that are isolated when
    the dataset loads (salt ions in molecules) and declares that a node an edit leaves isolated is deleted,
    which the oracle applies before predicting. Explainers' candidates are not touched here (manipulate
    on a candidate is a no-op): the edit stays in the input's frame and the oracle drops the node."""

    def process(self):
        self.dataset.isolated_nodes = "delete"
        for instance in self.dataset.instances:
            stripped, _ = drop_isolated(instance)
            if stripped is not instance:
                instance.data = stripped.data
                instance.node_features = stripped.node_features
                instance._nx_repr = None
        self.dataset._num_nodes = None
        self.dataset._num_nodes_values = None

    def process_instance(self, instance):
        pass
