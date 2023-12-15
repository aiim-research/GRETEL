import copy
import numpy as np

from src.core.explainer_base import Explainer
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric
from src.n_dataset.instances.graph import GraphInstance


class DCESearchExplainerWithRank(Explainer):

    # Todo improve configuration and default values
    def init(self):
        super().init()
        self._gd = GraphEditDistanceMetric()
        self.fold_id=-1        
        self.dist_mat = np.full((len(self.dataset.instances), len(self.dataset.instances)), -1)
        self.cls_mat = np.full((len(self.dataset.instances), len(self.dataset.instances)), -1)


    def explain(self, instance: GraphInstance):
        l_input_inst = self.oracle.predict(instance)

        # if the method does not find a counterfactual example returns the original graph
        min_counterfactual = instance

        if "distance_rank_index" in self.dataset.graph_features_map.keys():
            rank_index = self.dataset.graph_features_map["distance_rank_index"]   

            rank = [ int(x) for x in instance.graph_features[:,rank_index]]

            for value in rank:
                _inst = [x for x in self.dataset.instances if x.id == value][0]
                l_data_inst = self.oracle.predict(_inst)

                if l_data_inst != l_input_inst:
                    min_counterfactual = _inst
        
        # if manipulation data is not present, then use the legacy implementation of the dce
        else:
            for d_inst in self.dataset.instances:
                if self.cls_mat[instance.id,d_inst.id] == -1:
                    l_data_inst = self.oracle.predict(d_inst)
                    self.cls_mat[instance.id,d_inst.id] = (l_input_inst == l_data_inst)
                    self.cls_mat[d_inst.id,instance.id] = (l_input_inst == l_data_inst)

                if self.cls_mat[instance.id,d_inst.id] == 0:
                    if self.dist_mat[instance.id,d_inst.id] == -1:                
                        d_inst_dist = self._gd.evaluate(instance, d_inst, self.oracle)
                        self.dist_mat[instance.id,d_inst.id]=d_inst_dist
                        self.dist_mat[d_inst.id,instance.id]=d_inst_dist

                    d_inst_dist=self.dist_mat[instance.id,d_inst.id]
                    if (d_inst_dist < min_counterfactual_dist):                
                        min_counterfactual_dist = d_inst_dist
                        min_counterfactual = d_inst

        results = copy.deepcopy(min_counterfactual)
        results.id = instance.id

        return results





