import copy
import sys
from abc import ABC

from src.dataset.instances.graph import GraphInstance
from src.explainer.ensemble.aggregators.base import ExplanationAggregator
import numpy as np

from src.core.factory_base import get_instance_kvargs
from src.utils.cfg_utils import init_dflts_to_of

class ExplanationFrequency(ExplanationAggregator):

    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger

        dst_metric='src.evaluation.evaluation_metric_ged.GraphEditDistanceMetric'  

        #Check if the distance metric exist or build with its defaults:
        init_dflts_to_of(self.local_config, 'distance_metric', dst_metric)       

        if not 'ft' in self.local_config['parameters']:
            self.local_config['parameters']['ft'] = 3

        if not 'to_be_correct' in self.local_config['parameters']:
            self.local_config['parameters']['to_be_correct'] = False


    def init(self):
        super().init()       

        self.distance_metric = get_instance_kvargs(self.local_config['parameters']['distance_metric']['class'], 
                                                    self.local_config['parameters']['distance_metric']['parameters'])
        
        self.ft = self.local_config['parameters']['ft']
        self.if_correct = self.local_config['parameters']['to_be_correct']
        

    def aggregate(self, org_instance, explanations):
        org_lbl = self.oracle.predict(org_instance)

        # Getting the perturbation matrices of all the explanations that are valid counterfactuals
        explanations_perturbation_list = [exp.data for exp in explanations if (not self.if_correct or self.oracle.predict(exp) != org_lbl)]
        # Getting the frequency with which each edge is modified
        pert_freq = self.union_arrays(explanations_perturbation_list)

        # Getting all the edges above the frequency threshold
        sampled_edges = []
        for i in range(pert_freq.shape[0]):
            for j in range(pert_freq.shape[1]):
                if pert_freq[i,j] >= self.ft:
                    # sampled_edges.append([i, j])
                    pert_freq[i,j] = 1
                else:
                    pert_freq[i,j] = 0
                            
        cf_candidate = GraphInstance(id=org_instance.id,
                                    label=0,
                                    data=pert_freq,
                                    node_features=org_instance.node_features)
        
        for manipulator in org_instance._dataset.manipulators:
            manipulator._process_instance(cf_candidate)

        return cf_candidate
        
        # If no counterfactual was found return the original instance by convention
        # return copy.deepcopy(org_instance)
    

    def union_arrays(self, arrays):
        result = np.zeros_like(arrays[0])
        for arr in arrays:
            result += arr
        return result