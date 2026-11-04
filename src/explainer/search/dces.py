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
            return self.explain_projected(instance, input_label).counterfactual_instances[0]

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

    def explain_projected(self, instance, input_label):
        """Protocol v2 generator contract (src/explainer/future/utils/projection.py), decided by the generator
        alone (decoupled protocol): the full projection onto the input's nodes (the candidate's edges, node
        features and edge attributes) of the nearest dataset graph of another class whose projection the
        oracle classifies differently from the input, candidates ranked by the structural edit cost under
        the correspondence. Under the isolated-node policy "delete" a candidate no larger than the input
        always qualifies (its projection is the candidate itself). When none does, the nearest candidate
        itself (seed_kind "fallback", with its correspondence), a counterfactual by construction. Returns
        an explanation whose info carries seed_kind, correspondences and candidate_id."""
        from src.explainer.future.utils.projection import project, ranked_candidates
        from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
        from src.utils.isolated import materialize, policy_of
        candidates = [c for c in self.dataset.instances if self.oracle.predict(c) != input_label]
        ranked = ranked_candidates(instance, candidates, self.projection)
        info = {}
        seed = None
        for _, _, candidate, mapping in ranked:
            projected = project(instance, candidate, mapping, features=True)
            if self.oracle.predict(projected) != input_label:
                seed = projected
                # The seed in the input's frame; as a counterfactual on its own it is materialized by the
                # caller (GenerateMinimize) or measured with this correspondence.
                info = {"seed_kind": "projection", "candidate_id": candidate.id,
                        "correspondences": [materialize(projected, policy_of(self.dataset))[1]]}
                break
        if seed is None and ranked:
            _, _, candidate, mapping = ranked[0]
            seed = copy.deepcopy(candidate)
            info = {"seed_kind": "fallback", "candidate_id": candidate.id, "correspondences": [mapping]}
        if seed is None:
            seed = copy.deepcopy(instance)
            info = {"seed_kind": "none"}
        seed.id = instance.id
        explanation = LocalGraphCounterfactualExplanation(context=self.context, dataset=self.dataset,
                                                          oracle=self.oracle, explainer=self,
                                                          input_instance=instance, counterfactual_instances=[seed])
        explanation._info.update(info)
        return explanation
