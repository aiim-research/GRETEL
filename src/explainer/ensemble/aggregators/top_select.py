import copy
import sys
from abc import ABC

from src.core.explainer_base import Explainer
from src.explainer.ensemble.aggregators.base import ExplanationAggregator
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric
import numpy as np

from src.core.factory_base import get_instance_kvargs
from src.utils.cfg_utils import get_dflts_to_of, init_dflts_to_of, inject_dataset, inject_oracle, retake_oracle, retake_dataset

class ExplanationTopSelect(ExplanationAggregator):

    def init(self):
        super().init()

        self.distance_metric = get_instance_kvargs(self.local_config['parameters']['distance_metric']['class'], 
                                                    self.local_config['parameters']['distance_metric']['parameters'])

    def aggregate(self, org_instance, explanations):
        org_lbl = self.oracle.predict(org_instance)

        result = org_instance
        best_ged = sys.float_info.max
        for exp in explanations:
            exp_lbl = self.oracle.predict(exp)
            if exp_lbl != org_lbl:
                exp_dist = self.distance_metric.evaluate(org_instance, exp)
                if exp_dist < best_ged:
                    best_ged = exp_dist
                    result = exp

        return result
    
    def check_configuration(self):
        super().check_configuration()

        dst_metric='src.evaluation.evaluation_metric_ged.GraphEditDistanceMetric'  

        #Check if the distance metric exist or build with its defaults:
        init_dflts_to_of(self.local_config, 'distance_metric', dst_metric)