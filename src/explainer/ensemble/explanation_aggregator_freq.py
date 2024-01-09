import copy
import sys
from abc import ABC

from src.dataset.instances.graph import GraphInstance
from src.core.explainer_base import Explainer
from src.explainer.ensemble.explanation_aggregator_base import ExplanationAggregator
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric
from src.utils.samplers.bernoulli import Bernoulli
import numpy as np

from src.core.factory_base import get_instance_kvargs
from src.utils.cfg_utils import get_dflts_to_of, init_dflts_to_of, inject_dataset, inject_oracle, retake_oracle, retake_dataset


class ExplanationFrequency(ExplanationAggregator):

    def check_configuration(self):
        super().check_configuration()

        dst_metric='src.evaluation.evaluation_metric_ged.GraphEditDistanceMetric'  

        #Check if the distance metric exist or build with its defaults:
        init_dflts_to_of(self.local_config, 'distance_metric', dst_metric)

        if not 'ft' in self.local_config['parameters']:
            self.local_config['parameters']['ft'] = 3


    def init(self):
        super().init()

        self.distance_metric = get_instance_kvargs(self.local_config['parameters']['distance_metric']['class'], 
                                                    self.local_config['parameters']['distance_metric']['parameters'])
        
        self.ft = self.local_config['parameters']['ft']

        self.sampler = Bernoulli(1)


    def aggregate(self, org_instance, explanations):
        org_lbl = self.oracle.predict(org_instance)

        # Getting the perturbation matrices of all the explanations that are valid counterfactuals
        # explanations_perturbation_list = [(org_instance.data != exp.data).astype(int) for exp in explanations if self.oracle.predict(exp) != org_lbl]
        explanations_perturbation_list = [exp.data for exp in explanations if self.oracle.predict(exp) != org_lbl]
        # Getting the frequency with which each edge is modified
        pert_freq = self.union_arrays(explanations_perturbation_list)

        # normalized_A = pert_freq / pert_freq.sum(axis=1, keepdims=True)
        # samples = self.sampler.sample(org_instance, 
        #                               self.oracle,
        #                               **{'embedded_features': org_instance.node_features,
        #                                 'edge_probabilities': normalized_A})

        # Getting all the edges above the frequency threshold
        sampled_edges = []
        for i in range(pert_freq.shape[0]):
            for j in range(pert_freq.shape[1]):
                if pert_freq[i,j] >= self.ft:
                    # sampled_edges.append([i, j])
                    pert_freq[i,j] = 1
                else:
                    pert_freq[i,j] = 0
        # sampled_edges = np.array(sampled_edges)

        # If there are edges that meet the desired criteria
        # if len(sampled_edges) > 0:
        #     cf_cand_matrix = np.copy(org_instance.data)
        #     # switch on/off the sampled edges
        #     cf_cand_matrix[sampled_edges[:,0], sampled_edges[:,1]] = 1 - cf_cand_matrix[sampled_edges[:,0], sampled_edges[:,1]]
        #     cf_cand_matrix[sampled_edges[:,1], sampled_edges[:,0]] = 1 - cf_cand_matrix[sampled_edges[:,1], sampled_edges[:,0]]
        
        #     # build the counterfactaul candidates instance
        #     result = GraphInstance(id=org_instance.id,
        #                             label=0,
        #                             data=cf_cand_matrix,
        #                             node_features=org_instance.node_features)
            
        #     # if a counterfactual was found return that
        #     l_cf_cand = self.oracle.predict(result)
        #     if org_lbl != l_cf_cand:
        #         result.label = l_cf_cand
        #         return result
                    
        result = GraphInstance(id=org_instance.id,
                                    label=0,
                                    data=pert_freq,
                                    node_features=org_instance.node_features)
        
        return result
        
        # If no counterfactual was found return the original instance by convention
        # return copy.deepcopy(org_instance)
    

    def union_arrays(self, arrays):
        result = np.zeros_like(arrays[0])
        for arr in arrays:
            result += arr
        return result