import copy
import sys
from abc import ABC

from src.core.explainer_base import Explainer
from src.explainer.ensemble.explanation_aggregator_base import ExplanationAggregator
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric
import numpy as np

from src.core.factory_base import get_instance_kvargs
from src.utils.cfg_utils import get_dflts_to_of, init_dflts_to_of, inject_dataset, inject_oracle, retake_oracle, retake_dataset


class ExplanationUnion(ExplanationAggregator):

    def init(self):
        super().init()

        self.distance_metric = get_instance_kvargs(self.local_config['parameters']['distance_metric']['class'], 
                                                    self.local_config['parameters']['distance_metric']['parameters'])


    def aggregate(self, org_instance, explanations):
        # Getting the union of the adjacency matrices of all explanations
        explanations_A_list = [exp.data for exp in explanations]
        A_union = self.union_arrays(explanations_A_list)

        # cloning the first explanation
        result = copy.deepcopy(explanations[0])
        result.data = A_union

        return result
    

    def union_arrays(self, arrays):
        result = np.zeros_like(arrays[0], dtype=int)
        for arr in arrays:
            result |= arr
        return result
    
    
    def check_configuration(self):
        super().check_configuration()

        dst_metric='src.evaluation.evaluation_metric_ged.GraphEditDistanceMetric'  

        #Check if the distance metric exist or build with its defaults:
        init_dflts_to_of(self.local_config, 'distance_metric', dst_metric)