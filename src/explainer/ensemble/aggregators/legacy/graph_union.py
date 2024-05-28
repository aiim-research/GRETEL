from typing import List
import copy
import numpy as np

from src.dataset.instances.graph import GraphInstance
from src.dataset.instances.base import DataInstance
from src.explainer.ensemble.aggregators.base import ExplanationAggregator
from src.utils.utils import pad_adj_matrix


class ExplanationGraphUnion(ExplanationAggregator):

    def real_aggregate(self, instance: DataInstance, explanations: List[DataInstance]):
        # If the correctness filter is active then consider only the correct explanations in the list
        if self.correctness_filter:
            filtered_explanations = self.filter_correct_explanations(instance, explanations)
        else:
            # Consider all the explanations in the list
            filtered_explanations = explanations

        if len(filtered_explanations) < 1:
            return copy.deepcopy(instance)

        # Get the number of nodes of the bigger explanation instance
        max_dim = max([exp.data.shape[0] for exp in filtered_explanations])
        # Create an empty adjacency matrix with the size of the bigger explanation instance
        edge_freq_matrix = np.zeros((max_dim, max_dim))
        
        # Sum the padded explanation instances to the edge-frequency matrix
        for exp in filtered_explanations:
            edge_freq_matrix = np.add(edge_freq_matrix, pad_adj_matrix(exp.data, max_dim))

        # Create the union adjacency matrix with one in all the positions where the edge frequency is greater than 1
        union_matrix = np.where(edge_freq_matrix >= 1, 1, 0)
        # Create the aggregated explanation
        aggregated_explanation = GraphInstance(id=instance.id, label=1-instance.label, data=union_matrix)
        self.dataset.manipulate(aggregated_explanation)
        aggregated_explanation.label = self.oracle.predict(aggregated_explanation)

        return aggregated_explanation