from typing import List
import copy
import numpy as np

from src.dataset.instances.graph import GraphInstance
from src.dataset.instances.base import DataInstance
from src.explainer.ensemble.aggregators.base import ExplanationAggregator
from src.utils.utils import pad_adj_matrix


class ExplanationGraphFrequency(ExplanationAggregator):

    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger

        if 'frequency_threshold' not in self.local_config['parameters']:
            self.local_config['parameters']['frequency_threshold'] = 0.3

        if self.local_config['parameters']['frequency_threshold'] < 0:
            self.local_config['parameters']['frequency_threshold'] = 0
        elif self.local_config['parameters']['frequency_threshold'] > 1.0:
            self.local_config['parameters']['frequency_threshold'] = 1.0


    def init(self):
        super().init()

        self.freq_t = self.local_config['parameters']['frequency_threshold']
        

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

        # calculating the frequency threshold
        n_exp = len(filtered_explanations)
        freq_threshold = int(n_exp * self.freq_t)
        # In case the given threshold falls below 0 then default to the minimum value of 1 and produce the union
        if freq_threshold < 1:
            freq_threshold = 1

        # Create the union adjacency matrix with one in all the positions where the edge frequency is greater than the given threshold
        union_matrix = np.where(edge_freq_matrix >= freq_threshold, 1, 0)
        # Create the aggregated explanation
        aggregated_explanation = GraphInstance(id=instance.id, label=1-instance.label, data=union_matrix)
        self.dataset.manipulate(aggregated_explanation)
        aggregated_explanation.label = self.oracle.predict(aggregated_explanation)

        return aggregated_explanation
    