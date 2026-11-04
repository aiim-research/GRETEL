import copy
import sys

from src.core.explainer_base import Explainer
import numpy as np

from src.core.factory_base import get_instance_kvargs
from src.utils.cfg_utils import  init_dflts_to_of 

class DCESExplainer(Explainer):
    """The Distribution Compliant Explanation Search Explainer performs a search of 
    the minimum counterfactual instance in the original dataset instead of generating
    a new instance"""

    def check_configuration(self):
        super().check_configuration()

        dst_metric='src.evaluation.evaluation_metric_ged.GraphEditDistanceMetric'  

        #Check if the distance metric exist or build with its defaults:
        init_dflts_to_of(self.local_config, 'distance_metric', dst_metric)

        # Protocol v2 (opt-in, read without being written into the config): return the nearest candidate of
        # another class whose projection onto the input's nodes is a counterfactual, ranked by the
        # structural edit cost under the correspondence "index" or "assignment"
        # (src/explainer/future/utils/projection.py). Without it, DCE behaves as always.
        self.projection = self.local_config['parameters'].get('projection')


    def init(self):
        super().init()

        self.distance_metric = get_instance_kvargs(self.local_config['parameters']['distance_metric']['class'], 
                                                    self.local_config['parameters']['distance_metric']['parameters'])

    def explain(self, instance):
        input_label = self.oracle.predict(instance)

        if self.projection:
            from src.explainer.future.utils.projection import nearest_valid_projection
            candidates = [c for c in self.dataset.instances if self.oracle.predict(c) != input_label]
            projected, _ = nearest_valid_projection(instance, candidates, self.oracle, self.projection, input_label)
            result = copy.deepcopy(instance) if projected is None else projected
            result.id = instance.id
            return result

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

        # print("dce min_ctf -> " + str(self.oracle.predict(min_ctf)))
        result = copy.deepcopy(min_ctf)
        result.id = instance.id

        return result