import numpy as np


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

    # Get the difference in the number of nodes
    nodes_diff_count = abs(matrix_1.shape[0] - matrix_2.shape[0])

    # Get the shape of the matrices
    shape_matrix_1 = matrix_1.shape
    shape_matrix_2 = matrix_2.shape

    # Find the minimum dimensions of the matrices
    min_shape = (min(shape_matrix_1[0], shape_matrix_2[0]), min(shape_matrix_1[1], shape_matrix_2[1]))

    # Initialize an empty list to store the differences
    edges_diff = []

    # Iterate over the common elements of the matrices
    for i in range(min_shape[0]):
        for j in range(min_shape[1]):
            if matrix_1[i,j] != matrix_2[i,j]:
                edges_diff.append((i,j))

    # If the matrices have different shapes, loop through the remaining cells in
    # the larger matrix (the matrixes are square shaped)
    if shape_matrix_1 != shape_matrix_2:
        max_shape = np.maximum(shape_matrix_1, shape_matrix_2)

        for i in range(min_shape[0], max_shape[0]):
            for j in range(min_shape[1], max_shape[1]):
                if shape_matrix_1 > shape_matrix_2:
                    edge_val = matrix_1[i,j]
                else:
                    edge_val = matrix_2[i,j]

                # Only add non-zero cells to the list
                if edge_val != 0:  
                    edges_diff.append((i, j))

    edges_diff_count = len(edges_diff)
    if not directed:
        edges_diff_count /= 2

    return nodes_diff_count + edges_diff_count
