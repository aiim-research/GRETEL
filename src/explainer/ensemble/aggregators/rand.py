import copy
from typing import List

from src.dataset.instances.graph import GraphInstance
from src.explainer.ensemble.aggregators.base import ExplanationAggregator
import numpy as np

from src.core.factory_base import get_instance_kvargs
from src.utils.cfg_utils import init_dflts_to_of
from src.utils.utils import pad_features

class ExplanationRandom(ExplanationAggregator):

    def init(self):
        super().init()

        self.distance_metric = get_instance_kvargs(self.local_config['parameters']['distance_metric']['class'], 
                                                    self.local_config['parameters']['distance_metric']['parameters'])
        self.tries = 5
        self.k = 15

    def real_aggregate(self, org_instance: GraphInstance, explanations: List[GraphInstance]):
        org_lbl = self.oracle.predict(org_instance)

        max_dim = max([exp.data.shape[0] for exp in explanations])
        all_changes_matrix = np.zeros((max_dim, max_dim), dtype=int)

        for exp in explanations:
            # If the explanation is a correct counterfactual
            if org_lbl != self.oracle.predict(exp):

                if exp.data.shape[0] < max_dim:
                    exp = np.pad(exp, 
                                 pad_width=((0, max_dim - exp.data.shape[0]), (0, max_dim - exp.data.shape[0])), 
                                 mode='constant', 
                                 constant_values=0)
                # TODO: @mario this one breaks if the shapes of the two adj matrices aren't the same
                # it always breaks if the explainer is a search-based one
                # it will always break if the explainer adds/removes nodes


                changes = (org_instance.data != exp.data).astype(int)
                all_changes_matrix |= changes

        changed_edges = np.nonzero(all_changes_matrix)
        num_changed_edges = len(changed_edges[0])
        # if there are any changes, perform the random search
        if num_changed_edges:
            new_edges = np.array([[changed_edges[0][i], changed_edges[1][i]] for i in range(num_changed_edges)])
            # increase the number of random modifications
            for i in range(1, self.k):
                # how many attempts at a current modification level
                for _ in range(0, self.tries):
                    cf_cand_matrix = copy.deepcopy(org_instance.data)

                    # Padding the cf candidate
                    if cf_cand_matrix.shape[0] < max_dim:
                        cf_cand_matrix = np.pad(exp, 
                                                pad_width=((0, max_dim - cf_cand_matrix.shape[0]), (0, max_dim - cf_cand_matrix.shape[0])), 
                                                mode='constant', 
                                                constant_values=0)
                    
                    # sample according to perturbation_percentage
                    sample_index = np.random.choice(list(range(len(new_edges))), size=i, replace=False)
                    sampled_edges = new_edges[sample_index]
                    # switch on/off the sampled edges
                    cf_cand_matrix[sampled_edges[:,0], sampled_edges[:,1]] = 1 - cf_cand_matrix[sampled_edges[:,0], sampled_edges[:,1]]                
                    
                    # build the counterfactaul candidates instance
                    features = pad_features(org_instance.node_features, max_dim)
                    result = GraphInstance(id=org_instance.id, label=-1, data=cf_cand_matrix, node_features=features)
                    self.dataset.manipulate(result)
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