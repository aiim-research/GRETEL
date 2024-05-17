from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer
import numpy as np

class GraphEditDistanceMetric:
    """Provides a graph edit distance function for graphs where nodes are already matched, 
    thus eliminating the need of performing an NP-Complete graph matching.
    """

    def __init__(self, node_insertion_cost=1.0, node_deletion_cost=1.0, edge_insertion_cost=1.0,
                 edge_deletion_cost=1.0, undirected=True, config_dict=None) -> None:
        self._name = 'Graph_Edit_Distance'
        self._node_insertion_cost = node_insertion_cost
        self._node_deletion_cost = node_deletion_cost
        self._edge_insertion_cost = edge_insertion_cost
        self._edge_deletion_cost = edge_deletion_cost
        self.undirected = undirected
        

    def evaluate(self, instance_1 , instance_2 , oracle : Oracle=None, explainer : Explainer=None, dataset = None):
        # Implementation for numpy matrices
        A_g1 = instance_1.data
        A_g2 = instance_2.data

        # Get the difference in the number of nodes
        nodes_diff_count = abs(A_g1.shape[0] - A_g2.shape[0])

        # Get the shape of the matrices
        shape_A_g1 = A_g1.shape
        shape_A_g2 = A_g2.shape

        # Find the minimum dimensions of the matrices
        min_shape = (min(shape_A_g1[0], shape_A_g2[0]), min(shape_A_g1[1], shape_A_g2[1]))

        # Initialize an empty list to store the differences
        edges_diff = []

        # Iterate over the common elements of the matrices
        for i in range(min_shape[0]):
            for j in range(min_shape[1]):
                if A_g1[i,j] != A_g2[i,j]:
                    edges_diff.append((i,j))

        # If the matrices have different shapes, loop through the remaining cells in the larger matrix (the matrixes are square shaped)
        if shape_A_g1 != shape_A_g2:
            max_shape = np.maximum(shape_A_g1, shape_A_g2)

            for i in range(min_shape[0], max_shape[0]):
                for j in range(min_shape[1], max_shape[1]):
                    if shape_A_g1 > shape_A_g2:
                        edge_val = A_g1[i,j]
                    else:
                        edge_val = A_g2[i,j]

                    # Only add non-zero cells to the list
                    if edge_val != 0:  
                        edges_diff.append((i, j))

        edges_diff_count = len(edges_diff)
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

    