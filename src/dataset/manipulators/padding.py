

import numpy as np

from src.dataset.manipulators.base import BaseManipulator
from src.utils.utils import pad_adj_matrix, pad_features


class AdjacencyMatrixPadder(BaseManipulator):

    def process(self):
        max_num_nodes = np.max(self.dataset.num_nodes_values)
        for instance in self.dataset.instances:
            # Pad the adjacency matrix (instance.data) to be max_num_nodes x max_num_nodes
            instance.data = pad_adj_matrix(instance.data, max_num_nodes)
            instance.node_features = pad_features(instance.node_features, max_num_nodes)

        super().process()