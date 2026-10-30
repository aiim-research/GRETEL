from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer
import numpy as np

from src.utils.metrics.ged import matched_edge_diff_count

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
        

    def evaluate(self, instance_1 , instance_2 , oracle : Oracle=None, explainer : Explainer=None, dataset = None):
        # G1 = instance_1.graph
        # G2 = instance_2.graph

        # edit_distance = 0.0

        # for n in G1.nodes:
        #     if not G2.has_node(n):
        #         edit_distance += self._node_deletion_cost

        # for n in G2.nodes:
        #     if not G1.has_node(n):
        #         edit_distance += self._node_insertion_cost

        # for e in G1.edges:
        #     if not G2.has_edge(*e):
        #         edit_distance += self._edge_deletion_cost

        # for e in G2.edges:
        #     if not G1.has_edge(*e):
        #         edit_distance += self._edge_insertion_cost

        # return edit_distance
    
        # Implementation for numpy matrices
        A_g1 = instance_1.data
        A_g2 = instance_2.data

        # Bardh idea ----------------------------------------------------------

        # result = float(np.sum(np.absolute(A_g1 - A_g2)))
        # return result
        # ---------------------------------------------------------------------

        # Vectorised cell count, same result as the former cell-by-cell loop (src/utils/metrics/ged.py)
        nodes_diff_count = abs(A_g1.shape[0] - A_g2.shape[0])
        edges_diff_count = matched_edge_diff_count(A_g1, A_g2)
        if self.undirected:
            edges_diff_count /= 2

        return nodes_diff_count + edges_diff_count
    
    
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

    