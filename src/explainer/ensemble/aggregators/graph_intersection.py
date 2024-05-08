import copy
import numpy as np
from typing import List

from src.dataset.instances.base import DataInstance
from src.dataset.instances.graph import GraphInstance
from src.explainer.ensemble.aggregators.base import ExplanationAggregator
from src.utils.utils import pad_adj_matrix


class ExplanationGraphIntersection(ExplanationAggregator):

    def real_aggregate(self, instance: DataInstance, explanations: List[DataInstance]):
        # If the correctness filter is active then consider only the correct explanations in the list
        if self.correctness_filter:
            filtered_explanations = self.filter_correct_explanations(instance, explanations)
        else:
            # Consider all the explanations in the list
            filtered_explanations = explanations

        if len(filtered_explanations) < 1:
            return copy.deepcopy(instance)

        # Gettting the number of nodes of the bigger explanation instance
        max_dim = max([exp.data.shape[0] for exp in filtered_explanations])
        # Initializing the edge-frequency matrix with 1 in all positions
        edge_freq_matrix = np.ones((max_dim, max_dim))

        # Multiply the adj matrix of the explanations for the edge-frequency matrix
        # only positions that are 1 in all explanations will remain 1
        for exp in filtered_explanations:
            edge_freq_matrix = edge_freq_matrix * pad_adj_matrix(exp.data, max_dim)

        # Getting the intersection adjacency matrix
        intersection_matrix = np.where(edge_freq_matrix > 0, 1, 0)
        # Creating the aggregated explanation instance
        aggregated_explanation = GraphInstance(id=instance.id, data=intersection_matrix, label=1-instance.label)
        self.dataset.manipulate(aggregated_explanation)
        aggregated_explanation.label = self.oracle.predict(aggregated_explanation)

        return aggregated_explanation
       