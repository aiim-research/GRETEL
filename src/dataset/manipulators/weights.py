

import numpy as np

from src.dataset.manipulators.base import BaseManipulator


class EdgeWeights(BaseManipulator):
    
    def edge_info(self, instance):
        adj = instance.data
        # the weights need to be an array of real numbers with
        # length equal to the number of edges
        row_indices, col_indices = np.where(adj != 0)
        instance.edge_weights = adj[row_indices, col_indices]
        
        # return { "edge_weights": list(weights) }
        return { }