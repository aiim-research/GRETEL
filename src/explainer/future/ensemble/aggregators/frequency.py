from typing import List
import copy

from src.dataset.instances.graph import GraphInstance
from src.dataset.instances.base import DataInstance
from src.explainer.future.ensemble.aggregators.base import ExplanationAggregator
from src.utils.utils import pad_adj_matrix
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
import src.utils.explanations.functions as exp_tools


class ExplanationFrequency(ExplanationAggregator):

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
        

    def real_aggregate(self, explanations: List[LocalGraphCounterfactualExplanation]) -> LocalGraphCounterfactualExplanation:
        # calculating the frequency threshold
        input_inst = explanations[0].input_instance
        cf_instances = exp_tools.unpack_cf_instances(explanations)
        n_exp = len(cf_instances)

        freq_threshold = int(n_exp * self.freq_t)
        # In case the given threshold falls below 0 then default to the minimum value of 1 and produce the union
        if freq_threshold < 1:
            freq_threshold = 1

        # Get the number of nodes of the bigger explanation instance
        max_dim = max(input_inst.data.shape[0], max([cf.data.shape[0] for cf in cf_instances]))

        # Get all the changes in all explanations
        mod_edges, _, mod_freq_matrix = self.get_all_edge_differences(input_inst, cf_instances)
        # Apply to the original matrix those changes that where performed by all explanations
        intersection_matrix = pad_adj_matrix(copy.deepcopy(input_inst.data), max_dim)
        for edge in mod_edges:
            if mod_freq_matrix[edge[0], edge[1]] >= freq_threshold:
                intersection_matrix[edge[0], edge[1]] = abs(intersection_matrix[edge[0], edge[1]] - 1 )

                # If the graphs are undirected
                if not input_inst.is_directed:
                    # Assign to the symetrical edge the same value than to the original edge
                    intersection_matrix[edge[1], edge[0]] = intersection_matrix[edge[0], edge[1]] # The original edge was already modified

        # Create the aggregated explanation
        aggregated_instance = GraphInstance(id=input_inst.id, label=1-input_inst.label, data=intersection_matrix)
        self.dataset.manipulate(aggregated_instance)
        aggregated_instance.label = self.oracle.predict(aggregated_instance)

        aggregated_explanation = LocalGraphCounterfactualExplanation(context=self.context,
                                                                    dataset=self.dataset,
                                                                    oracle=self.oracle,
                                                                    explainer=None, # Will be added later by the ensemble
                                                                    input_instance=input_inst,
                                                                    counterfactual_instances=[aggregated_instance])

        return aggregated_explanation
    