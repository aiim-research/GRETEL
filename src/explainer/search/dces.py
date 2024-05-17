import copy
import sys
import numpy as np

from src.core.explainer_base import Explainer
from src.core.factory_base import get_instance_kvargs
from src.utils.cfg_utils import  init_dflts_to_of 
from src.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation

class DCESExplainer(Explainer):
    """The Distribution Compliant Explanation Search Explainer performs a search of 
    the minimum counterfactual instance in the original dataset instead of generating
    a new instance"""

    def check_configuration(self):
        super().check_configuration()

        dst_metric='src.utils.metrics.ged.GraphEditDistanceMetric'  

        #Check if the distance metric exist or build with its defaults:
        init_dflts_to_of(self.local_config, 'distance_metric', dst_metric)


    def init(self):
        super().init()

        self.distance_metric = get_instance_kvargs(self.local_config['parameters']['distance_metric']['class'], 
                                                    self.local_config['parameters']['distance_metric']['parameters'])

    def explain(self, instance):
        input_label = self.oracle.predict(instance)

        # if the method does not find a counterfactual example returns the original graph
        min_ctf = instance

        # Iterating over all the instances of the dataset
        min_ctf_dist = sys.float_info.max
        for ctf_candidate in self.dataset.instances:
            candidate_label = self.oracle.predict(ctf_candidate)

            if input_label != candidate_label:
                ctf_distance = self.distance_metric.evaluate(instance, ctf_candidate, self.oracle)
                
                if ctf_distance < min_ctf_dist:
                    min_ctf_dist = ctf_distance
                    min_ctf = ctf_candidate

        # A Local Graph Counterfactual Explanation is created as the return of the method
        result = LocalGraphCounterfactualExplanation(explainer_class=self.name,
                                                     input_instance=instance,
                                                     counterfactual_instances=[copy.deepcopy(min_ctf)]
                                                     )
        return result