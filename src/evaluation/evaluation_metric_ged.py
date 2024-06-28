import numpy as np

from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer
from src.utils.metrics.ged import graph_edit_distance_metric

class GraphEditDistanceMetric(EvaluationMetric):
    """Provides a graph edit distance function for graphs where nodes are already matched, 
    thus eliminating the need of performing an NP-Complete graph matching.
    """

    def __init__(self, node_insertion_cost=1.0, node_deletion_cost=1.0, edge_insertion_cost=1.0,
                 edge_deletion_cost=1.0, undirected=True, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Graph_Edit_Distance'
        self._node_insertion_cost = node_insertion_cost
        self._node_deletion_cost = node_deletion_cost
        self._edge_insertion_cost = edge_insertion_cost
        self._edge_deletion_cost = edge_deletion_cost
        self.undirected = undirected

    def evaluate(self, instance , explanation , oracle : Oracle=None, explainer : Explainer=None, dataset = None):
        instance_2 = explanation.top
        return graph_edit_distance_metric(
            instance.data,
            instance_2.data,
            instance.directed and instance_2.directed
        )
    
    def aggregate(self, measure_list, instances_correctness_list=None):
        # If no correctness list is provided aggregate all the measures
        if instances_correctness_list is None:
            return super().aggregate(measure_list, instances_correctness_list)
        else: # If correctness list is provided then aggregate only the measures of the correct instances
            filtered_measure_list = [item for item, flag in zip(measure_list, instances_correctness_list) if flag == 1]

            # Avoid aggregating an empty list
            if len(filtered_measure_list) > 0:
                return np.mean(filtered_measure_list), np.std(filtered_measure_list)
            else:
                return 0.0, 0.0

    