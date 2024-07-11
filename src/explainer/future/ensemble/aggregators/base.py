from copy import deepcopy
from typing import List
import numpy as np
import sys

from src.core.factory_base import get_instance_kvargs
from src.core.oracle_base import Oracle
from src.dataset.dataset_base import Dataset
from src.core.configurable import Configurable
from src.dataset.instances.base import DataInstance
from src.dataset.instances.graph import GraphInstance
from src.explainer.future.ensemble.aggregators.nodes.base import NodeFeatureAggregator
from src.utils.cfg_utils import init_dflts_to_of, inject_dataset, inject_oracle, retake_oracle, retake_dataset
from src.utils.utils import pad_adj_matrix
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.explainer.future.ensemble.aggregators.filters.base import ExplanationFilter

class ExplanationAggregator(Configurable):

    def check_configuration(self):
        super().check_configuration()
        
        if 'node_feature_aggregator' not in self.local_config['parameters']:
            init_dflts_to_of(self.local_config,
                             'node_feature_aggregator',
                             'src.explainer.future.ensemble.aggregators.nodes.average.AverageAggregator')
            
        if 'explanation_filter' not in self.local_config['parameters']:
            init_dflts_to_of(self.local_config,
                             'explanation_filter',
                             'src.explainer.future.ensemble.aggregators.filters.correctness.CorrectnessFilter')
            

    def init(self):
        self.dataset: Dataset = retake_dataset(self.local_config)
        self.oracle: Oracle = retake_oracle(self.local_config)
        
        # Injecting oracle and dataset in the ancillary classes
        inject_dataset(self.local_config['parameters']['node_feature_aggregator'], self.dataset)
        inject_oracle(self.local_config['parameters']['node_feature_aggregator'], self.oracle)

        inject_dataset(self.local_config['parameters']['explanation_filter'], self.dataset)
        inject_oracle(self.local_config['parameters']['explanation_filter'], self.oracle)
        
        
        self.node_feature_aggregator: NodeFeatureAggregator = get_instance_kvargs(self.local_config['parameters']['node_feature_aggregator']['class'],
                                                                                  {'context':self.context,'local_config': self.local_config['parameters']['node_feature_aggregator']})
        
        self.explanation_filter: ExplanationFilter = get_instance_kvargs(self.local_config['parameters']['explanation_filter']['class'],
                                                                                  {'context':self.context,'local_config': self.local_config['parameters']['explanation_filter']})

        super().init()


    def aggregate(self, explanations: List[LocalGraphCounterfactualExplanation]) -> LocalGraphCounterfactualExplanation:
        # First let's filter the base explanations to ensure only those with the desired properties are used
        filtered_explanations = self.explanation_filter.filter(explanations)

        if len(filtered_explanations) < 1:
            # If there was no any base correct explanation then return an instance containing the input instance as explanation
            incorrect_exp = explanations[0]
            default_explanation = LocalGraphCounterfactualExplanation(context=incorrect_exp.context,
                                                                      dataset=incorrect_exp.dataset,
                                                                      oracle=incorrect_exp.oracle,
                                                                      explainer=incorrect_exp.explainer,
                                                                      input_instance=incorrect_exp.input_instance,
                                                                      counterfactual_instances=[incorrect_exp.input_instance])
            return default_explanation
        
        # we need to combine:
        # 1) graph features
        # First an aggregated explanation is produced
        aggregated_explanation = self.real_aggregate(explanations=filtered_explanations)
        # 2) node features
        updated_cf_insts = self.node_feature_aggregator.aggregate(aggregated_explanation, base_explanations=filtered_explanations)
        aggregated_explanation.counterfactual_instances = updated_cf_insts
        # 3) edge features
        
        return aggregated_explanation
            
            
    def real_aggregate(self, explanations: List[LocalGraphCounterfactualExplanation]) -> LocalGraphCounterfactualExplanation:
        # This is the method where the aggregation takes place. It should be implemented by the child classes
        pass
    

    def get_edge_differences(self, instance: DataInstance, cf_instance: DataInstance):
        # Summing the two adjacency matrices (the metrices need to have the same size) edges that appear only in one of the two instances are the different ones
        edge_freq_matrix = np.add(instance.data, cf_instance.data)
        diff_matrix = np.where(edge_freq_matrix == 1, 1, 0)
        diff_number = np.count_nonzero(diff_matrix)

        if instance.directed:
            filtered_diff_number = int(diff_number)
        else:
            filtered_diff_number = int(diff_number/2)

        return filtered_diff_number, diff_matrix
    

    def get_all_edge_differences(self, instance: DataInstance, explanations: List[DataInstance]):
        # Getting the max explanation instance dimension and padding the original instance
        max_dim = max(instance.data.shape[0], max([exp.data.shape[0] for exp in explanations]))
        padded_inst_adj_matrix = pad_adj_matrix(instance.data, max_dim)

        padded_instance = GraphInstance(id=instance.id, data=padded_inst_adj_matrix, label=instance.label, directed=instance.directed)
        self.dataset.manipulate(padded_instance)

        # Creating a matrix with the edges that were changed in any explanation and in how many explanations they were modified
        edge_change_freq_matrix = np.zeros((max_dim, max_dim))
        min_changes = sys.maxsize
        for exp in explanations:
            exp_changes_num, exp_changes_mat = self.get_edge_differences(padded_instance, pad_adj_matrix(exp.data, max_dim))

            # Aggregating the edges that were modified in each explanation
            edge_change_freq_matrix = np.add(edge_change_freq_matrix, exp_changes_mat)

            # Keeping what was the minimum number of changes performed by any explanation
            if exp_changes_num < min_changes:
                min_changes = exp_changes_num

        # Get the positions of the edge_change_freq
        edges = [(row, col) for row, col in zip(*np.where(edge_change_freq_matrix))]

        # If we are working with directed graphs
        if instance.directed:
            filtered_edges = edges
        else: # if we are working with undirected graphs
            filtered_edges = []
            for x, y in edges:
                if (y,x) not in filtered_edges:
                    filtered_edges.append((x,y))

        return filtered_edges, min_changes, edge_change_freq_matrix       