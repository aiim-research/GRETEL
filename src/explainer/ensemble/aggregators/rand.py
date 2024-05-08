import copy
from typing import List
import numpy as np
import random

from src.dataset.instances.base import DataInstance
from src.dataset.instances.graph import GraphInstance
from src.explainer.ensemble.aggregators.base import ExplanationAggregator
from src.utils.utils import pad_adj_matrix, pad_features

class ExplanationRandom(ExplanationAggregator):

    def check_configuration(self):
        super().check_configuration()

        if 'runs' not in self.local_config['parameters']:
            self.local_config['parameters']['runs'] = 5


    def init(self):
        super().init()

        self.runs = self.local_config['parameters']['runs']


    def real_aggregate(self, instance: DataInstance, explanations: List[DataInstance]):
        # If the correctness filter is active then consider only the correct explanations in the list
        if self.correctness_filter:
            filtered_explanations = self.filter_correct_explanations(instance, explanations)
        else:
            # Consider all the explanations in the list
            filtered_explanations = explanations

        if len(filtered_explanations) < 1:
            return copy.deepcopy(instance)
        
        # Getting the label of the original instance
        inst_lbl = self.oracle.predict(instance)

        change_edges, min_changes, change_freq_matrix = self.get_all_edge_differences(instance, filtered_explanations)

        # Perform r runs repeating the random search process
        # aggregated_explanation = copy.deepcopy(instance)
        for i in range(0, self.runs):
            # The working matrix for each run is a new copy of the instance adjacency matrix
            adj_matrix = copy.deepcopy(instance.data)
            # Randomly sample a number of edges equivalent to the smallest base explanation
            sampled_edges = random.sample(change_edges, min_changes)

            # Try to modified the chosen edges one by one until a counterfactual is found
            for edge in sampled_edges:
                adj_matrix[edge[0], edge[1]] = abs( adj_matrix[edge[0], edge[1]] - 1 )

                # Creating an instance with the modified adjacency matrix
                aggregated_explanation = GraphInstance(id=instance.id,
                                                       label=0,
                                                       data=adj_matrix)
                self.dataset.manipulate(aggregated_explanation)

                # Predicting the label of the instance
                exp_lbl = self.oracle.predict(aggregated_explanation)
                aggregated_explanation.label = exp_lbl

                # If a counterfactual has been found return it
                if exp_lbl != inst_lbl:
                    return aggregated_explanation

        # If no counterfactual was found, return the original instance
        return copy.deepcopy(instance)
