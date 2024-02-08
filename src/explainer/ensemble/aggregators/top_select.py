import copy
import sys
from abc import ABC
from typing import List

from src.core.explainer_base import Explainer
from src.dataset.instances.graph import GraphInstance
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

    def real_aggregate(self, org_instance: GraphInstance, explanations: List[GraphInstance]):
        # Getting the label of the original instance
        org_lbl = self.oracle.predict(org_instance)

        # Initializing the result with the original instance which will be returned if a 
        # counterfactual is not found
        result = org_instance
        min_ged = sys.maxsize

        # Iterate over the base explanations looking for a correct counterfactual with the lowest GED
        for exp in explanations:
            if self.oracle.predict(exp) != org_lbl: # If the explanation is correct
                exp_ged = self.distance_metric.evaluate(org_instance, exp) # Get the counterfactual GED

                # If the GED is lower that the best counterfactual so far then replace it 
                # with current counterfactual
                if (exp_ged < min_ged):
                    result = exp
                    min_ged = exp_ged

        # Return the explanation
        return result

    
    def check_configuration(self):
        super().check_configuration()

        dst_metric='src.evaluation.evaluation_metric_ged.GraphEditDistanceMetric' 

        #Check if the distance metric exist or build with its defaults:
        init_dflts_to_of(self.local_config, 'distance_metric', dst_metric)