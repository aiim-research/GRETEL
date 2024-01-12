import copy
import sys
from abc import ABC

from src.dataset.instances.graph import GraphInstance
from src.core.explainer_base import Explainer
from src.explainer.ensemble.aggregators.base import ExplanationAggregator
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric
import numpy as np

from src.core.factory_base import get_instance_kvargs
from src.utils.cfg_utils import get_dflts_to_of, init_dflts_to_of, inject_dataset, inject_oracle, retake_oracle, retake_dataset

class ExplanationRandom(ExplanationAggregator):

    def init(self):
        super().init()

        self.distance_metric = get_instance_kvargs(self.local_config['parameters']['distance_metric']['class'], 
                                                    self.local_config['parameters']['distance_metric']['parameters'])
        self.tries = 5

    def aggregate(self, org_instance, explanations):
        org_lbl = self.oracle.predict(org_instance)


        all_changes_matrix = np.zeros(org_instance.data.shape, dtype=int)
        k = 15

        for exp in explanations:
            # Get the oracle's prediction for the explanation
            exp_lbl = self.oracle.predict(exp)

            # If the explanation is a correct counterfactual
            if org_lbl != exp_lbl:
                changes = (org_instance.data != exp.data).astype(int)
                all_changes_matrix |= changes

        changed_edges = np.nonzero(all_changes_matrix)
        num_changed_edges = len(changed_edges[0])
        new_edges = [[changed_edges[0][i], changed_edges[1][i]] for i in range(num_changed_edges)]
        new_edges = np.array(new_edges)

        # increase the number of random modifications
        for i in range(1, k):
            # how many attempts at a current modification level
            for j in range(0, self.tries):
                cf_cand_matrix = np.copy(org_instance.data)
                # sample according to perturbation_percentage
                sample_index = np.random.choice(list(range(len(new_edges))), size=i)
                sampled_edges = new_edges[sample_index]

                # switch on/off the sampled edges
                cf_cand_matrix[sampled_edges[:,0], sampled_edges[:,1]] = 1 - cf_cand_matrix[sampled_edges[:,0], sampled_edges[:,1]]
                cf_cand_matrix[sampled_edges[:,1], sampled_edges[:,0]] = 1 - cf_cand_matrix[sampled_edges[:,1], sampled_edges[:,0]]
            
                # build the counterfactaul candidates instance
                result = GraphInstance(id=org_instance.id,
                                       label=0,
                                       data=cf_cand_matrix,
                                       node_features=org_instance.node_features)
                
                for manipulator in org_instance._dataset.manipulators:
                    manipulator._process_instance(result)
                
                # if a counterfactual was found return that
                l_cf_cand = self.oracle.predict(result)
                if org_lbl != l_cf_cand:
                    result.label = l_cf_cand
                    return result
        
        # If no counterfactual was found return the original instance by convention
        return copy.deepcopy(org_instance)
    
    def check_configuration(self):
        super().check_configuration()

        dst_metric='src.evaluation.evaluation_metric_ged.GraphEditDistanceMetric'  

        #Check if the distance metric exist or build with its defaults:
        init_dflts_to_of(self.local_config, 'distance_metric', dst_metric)