import copy
import sys

from src.core.explainer_base import Explainer
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric
import numpy as np


class DCESearchExplainer(Explainer):
    """The Distribution Compliant Explanation Search Explainer performs a search of 
    the minimum counterfactual instance in the original dataset instead of generating
    a new instance"""

    '''def __init__(self, id, instance_distance_function : EvaluationMetric, config_dict=None) -> None:
        super().__init__(id, config_dict)
        self._gd = instance_distance_function
        self._name = 'DCESearchExplainer'''

    def init(self):
        super().init()
        self._gd = GraphEditDistanceMetric()
        self.fold_id=-1        
        self.dist_mat = np.full((len(self.dataset.instances), len(self.dataset.instances)), -1)
        self.cls_mat = np.full((len(self.dataset.instances), len(self.dataset.instances)), -1)


    def explain(self, instance):
        l_input_inst = self.oracle.predict(instance)

        # if the method does not find a counterfactual example returns the original graph
        min_counterfactual = instance
        
        min_counterfactual_dist = sys.float_info.max
        
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
        '''result = DataInstance(min_counterfactual.id)
        result.graph = min_counterfactual.graph
        result.max_n_nodes = min_counterfactual.max_n_nodes
        result._np_array = min_counterfactual._np_array
        result.graph_dgl = min_counterfactual.graph_dgl
        result.n_node_types = min_counterfactual.n_node_types'''

        return results





