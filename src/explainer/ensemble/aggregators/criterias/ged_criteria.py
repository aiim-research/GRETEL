import numpy as np

from src.dataset.instances.graph import GraphInstance
from src.explainer.ensemble.aggregators.criterias.base_criteria import BaseCriteria
from src.explainer.ensemble.aggregators.criterias.gain_direction import GainDirection


class GraphEditDistanceCriteria(BaseCriteria[GraphInstance]):
    def gain_direction(self):
        return GainDirection.MINIMIZE

    def calculate(
        self,
        first_instance: GraphInstance,
        second_instance: GraphInstance,
    ) -> float:
        # Implementation for numpy matrices
        A_g1 = first_instance
        A_g2 = second_instance

        # Get the difference in the number of nodes
        nodes_diff_count = abs(A_g1.shape[0] - A_g2.shape[0])

        # Get the shape of the matrices
        shape_A_g1 = A_g1.shape
        shape_A_g2 = A_g2.shape

        # Find the minimum dimensions of the matrices
        min_shape = (
            min(shape_A_g1[0], shape_A_g2[0]),
            min(shape_A_g1[1], shape_A_g2[1]),
        )

        # Initialize an empty list to store the differences
        edges_diff = []

        # Iterate over the common elements of the matrices
        for i in range(min_shape[0]):
            for j in range(min_shape[1]):
                if A_g1[i, j] != A_g2[i, j]:
                    edges_diff.append((i, j))

        # If the matrices have different shapes, loop through the remaining cells in the larger matrix (the matrixes are square shaped)
        if shape_A_g1 != shape_A_g2:
            max_shape = np.maximum(shape_A_g1, shape_A_g2)

            for i in range(min_shape[0], max_shape[0]):
                for j in range(min_shape[1], max_shape[1]):
                    if shape_A_g1 > shape_A_g2:
                        edge_val = A_g1[i, j]
                    else:
                        edge_val = A_g2[i, j]

                    # Only add non-zero cells to the list
                    if edge_val != 0:
                        edges_diff.append((i, j))

        edges_diff_count = len(edges_diff)
        if self.undirected:
            edges_diff_count /= 2

        return nodes_diff_count + edges_diff_count
