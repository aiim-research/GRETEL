from typing import List


from src.dataset.instances.graph import GraphInstance
from src.explainer.ensemble.aggregators.base import ExplanationAggregator
import numpy as np

from src.utils.utils import pad_adj_matrix

class ExplanationUnion(ExplanationAggregator):

    def real_aggregate(self, instance: GraphInstance, explanations: List[GraphInstance]):
        max_dim = max([exp.data.shape[0] for exp in explanations])
        edge_freq_matrix = np.zeros((max_dim, max_dim))
        
        for exp in explanations:
            edge_freq_matrix = np.add(edge_freq_matrix, pad_adj_matrix(exp.data, max_dim))

        adj = np.where(edge_freq_matrix >= 1, 1, 0)

        return GraphInstance(id=instance.id, label=1-instance.label, data=adj)