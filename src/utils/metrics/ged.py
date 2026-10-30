import numpy as np

from src.core.explainer_base import Explainer
from src.core.oracle_base import Oracle
from src.dataset.instances.graph import GraphInstance

class GraphEditDistanceMetric:
    def evaluate(
        self,
        instance_1: GraphInstance,
        instance_2: GraphInstance,
        oracle: Oracle=None,
        explainer: Explainer=None,
        dataset=None,
    ) -> float:
        return graph_edit_distance_metric(
            instance_1.data,
            instance_2.data,
            instance_1.directed and instance_2.directed
        )



def graph_edit_distance_metric(matrix_1: np.ndarray, matrix_2: np.ndarray, directed: bool) -> float:
    """
    Provides a graph edit distance function for graphs where nodes are already matched, 
    thus eliminating the need of performing an NP-Complete graph matching.

    Args:
    - matrix_1: The adjacency matrix of the first graph
    - matrix_2: The adjacency matrix of the second graph
    - directed: A boolean indicating whether the graphs are directed or not

    Returns:
    - The graph edit distance between the two graphs
    """

    edges_diff_count = matched_edge_diff_count(matrix_1, matrix_2)
    if not directed:
        edges_diff_count /= 2

    return abs(matrix_1.shape[0] - matrix_2.shape[0]) + edges_diff_count


def matched_edge_diff_count(matrix_1: np.ndarray, matrix_2: np.ndarray) -> int:
    """Number of adjacency cells that differ between two node-matched graphs, vectorised (the cell-by-cell
    Python loop it replaces made DCE's scan of the training set take a minute per instance on graphs of
    200 nodes). Same count as that loop, including its convention for matrices of different sizes: the
    common top-left block cell by cell, plus the non-zero cells of the larger matrix's bottom-right block
    beyond the smaller one (the side strips are not counted)."""
    m0 = min(matrix_1.shape[0], matrix_2.shape[0])
    m1 = min(matrix_1.shape[1], matrix_2.shape[1])
    count = int(np.count_nonzero(matrix_1[:m0, :m1] != matrix_2[:m0, :m1]))
    if matrix_1.shape != matrix_2.shape:
        larger = matrix_1 if matrix_1.shape > matrix_2.shape else matrix_2
        count += int(np.count_nonzero(larger[m0:, m1:]))
    return count
