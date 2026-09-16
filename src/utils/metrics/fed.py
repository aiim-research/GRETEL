import numpy as np

from src.core.explainer_base import Explainer
from src.core.oracle_base import Oracle
from src.dataset.instances.graph import GraphInstance
from scipy.optimize import linear_sum_assignment

class FeatureEditDistanceMetric:
    def evaluate(
        self,
        instance_1: GraphInstance,
        instance_2: GraphInstance,
        oracle: Oracle=None,
        explainer: Explainer=None,
        dataset=None,
    ) -> float:
        return feature_edit_distance_metric(
            instance_1.data,
            instance_2.data,
            instance_1.directed and instance_2.directed
        )



def feature_edit_distance_metric(matrix_1: np.ndarray, matrix_2: np.ndarray, directed: bool) -> float:
    """
    Calculate the feature edit distance metric between two feature matrices.

    Parameters:
        matrix_1 (np.ndarray): Feature matrix of graph 1 (nodes x features).
        matrix_2 (np.ndarray): Feature matrix of graph 2 (nodes x features).
        directed (bool): Whether the metric is directed (order of nodes matters).

    Returns:
        float: The feature edit distance metric.
    """
    # Ensure matrices have the same shape
    if matrix_1.shape != matrix_2.shape:
        raise ValueError("Feature matrices must have the same shape.")

    # Calculate feature differences
    if directed:
        distance = np.sum(np.linalg.norm(matrix_1 - matrix_2, axis=1))
    else:
        # Compute the pairwise distances between all node features
        pairwise_distances = np.linalg.norm(
            matrix_1[:, None, :] - matrix_2[None, :, :], axis=2
        )
        
        # Solve the optimal assignment problem
        row_ind, col_ind = linear_sum_assignment(pairwise_distances)
        distance = pairwise_distances[row_ind, col_ind].sum()

    return distance
