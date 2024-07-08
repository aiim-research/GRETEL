import copy
from typing import List
import numpy as np
import random

from src.dataset.instances.base import DataInstance
from src.dataset.instances.graph import GraphInstance
from src.explainer.future.ensemble.aggregators.base import ExplanationAggregator
from src.utils.utils import pad_adj_matrix, pad_features
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
import src.utils.explanations.functions as exp_tools

class ExplanationRandom(ExplanationAggregator):

    def check_configuration(self):
        super().check_configuration()

        if 'runs' not in self.local_config['parameters']:
            self.local_config['parameters']['runs'] = 5


    def init(self):
        super().init()

        self.runs = self.local_config['parameters']['runs']


    def real_aggregate(self, explanations: List[LocalGraphCounterfactualExplanation]) -> LocalGraphCounterfactualExplanation:
        # calculating the frequency threshold
        input_inst = explanations[0].input_instance
        cf_instances = exp_tools.unpack_cf_instances(explanations)
        # n_exp = len(cf_instances)

        # Getting the label of the original instance
        inst_lbl = self.oracle.predict(input_inst)

        change_edges, min_changes, change_freq_matrix = self.get_all_edge_differences(input_inst, cf_instances)

        # Perform r runs repeating the random search process
        # aggregated_explanation = copy.deepcopy(instance)
        for i in range(0, self.runs):
            # The working matrix for each run is a new copy of the instance adjacency matrix
            adj_matrix = copy.deepcopy(input_inst.data)
            # Randomly sample a number of edges equivalent to the smallest base explanation
            sampled_edges = random.sample(change_edges, min_changes)

            # Try to modified the chosen edges one by one until a counterfactual is found
            for edge in sampled_edges:
                adj_matrix[edge[0], edge[1]] = abs( adj_matrix[edge[0], edge[1]] - 1 )

                # Creating an instance with the modified adjacency matrix
                aggregated_instance = GraphInstance(id=input_inst.id,
                                                       label=0,
                                                       data=adj_matrix)
                self.dataset.manipulate(aggregated_instance)
                # Predicting the label of the instance
                aggregated_instance.label = self.oracle.predict(aggregated_instance)

                # If a counterfactual has been found return it
                if aggregated_instance.label != inst_lbl:
                    aggregated_explanation = LocalGraphCounterfactualExplanation(context=self.context,
                                                                    dataset=self.dataset,
                                                                    oracle=self.oracle,
                                                                    explainer=None, # Will be added later by the ensemble
                                                                    input_instance=input_inst,
                                                                    counterfactual_instances=[aggregated_instance])

                    return aggregated_explanation

        # If no counterfactual was found, return the original instance
        no_explanation = LocalGraphCounterfactualExplanation(context=self.context,
                                                             dataset=self.dataset,
                                                             oracle=self.oracle,
                                                             explainer=None, # Will be added later by the ensemble
                                                             input_instance=input_inst,
                                                             counterfactual_instances=[copy.deepcopy(input_inst)])
        return no_explanation
