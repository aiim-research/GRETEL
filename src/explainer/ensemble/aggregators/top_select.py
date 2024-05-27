import copy
import sys
from typing import List

from src.dataset.instances.base import DataInstance
from src.explainer.ensemble.aggregators.base import ExplanationAggregator
from src.core.factory_base import get_instance_kvargs
from src.utils.cfg_utils import get_dflts_to_of, init_dflts_to_of, inject_dataset, inject_oracle, retake_oracle, retake_dataset
from src.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation

class ExplanationTopSelect(ExplanationAggregator):

    def check_configuration(self):
        super().check_configuration()

        dst_metric='src.utils.metrics.ged.GraphEditDistanceMetric' 

        #Check if the distance metric exist or build with its defaults:
        init_dflts_to_of(self.local_config, 'distance_metric', dst_metric)


    def init(self):
        super().init()

        self.distance_metric = get_instance_kvargs(self.local_config['parameters']['distance_metric']['class'], 
                                                    self.local_config['parameters']['distance_metric']['parameters'])
        

    def real_aggregate(self, explanations: List[LocalGraphCounterfactualExplanation]) -> LocalGraphCounterfactualExplanation:
        # Get the input instance from any explanation
        instance = explanations[0].input_instance

        # Initializing the result with the original instance which will be returned if a 
        # counterfactual is not found
        top_exp = instance
        top_exp_ged = sys.maxsize
        top_exp_obj = explanations[0]

        # Iterate over the base explanations looking for a correct counterfactual with the lowest GED
        # Note that if one of the base methods returned the original instance this is going to be selected as the GED is zero
        for exp_obj in explanations:
            for exp in exp_obj.counterfactual_instances:
                exp_ged = self.distance_metric.evaluate(instance, exp) # Get the counterfactual GED

                # If the GED is lower that the best counterfactual so far then replace it with current counterfactual.
                # We are explicitly avoiding to take the explanations that are the orginal instance, as that is the default value
                if (exp_ged < top_exp_ged and exp_ged >= 1):
                    top_exp = exp
                    top_exp_ged = exp_ged
                    top_exp_obj = exp_obj

        # Return the explanation
        result = top_exp_obj
        result.counterfactual_instances = [top_exp]
        return result