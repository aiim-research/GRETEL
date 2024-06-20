from typing import Callable

import numpy as np


def __min_max_normalize(
    criteria_matrix: np.ndarray,
) -> np.ndarray:
    min_vals = np.min(criteria_matrix, axis=0)
    max_vals = np.max(criteria_matrix, axis=0)
    normalized_matrix = (criteria_matrix - min_vals) / (max_vals - min_vals)
    return normalized_matrix


def __find_non_dominated_rows(
    criteria_matrix: np.ndarray,
    gain_directions: np.ndarray,
) -> np.ndarray:
    criteria_matrix_normalized = criteria_matrix * gain_directions
    num_rows = criteria_matrix.shape[0]
    non_dominated_indices = []
    for i in range(num_rows):
        dominated = False
        for j in range(num_rows):
            row1 = criteria_matrix_normalized[i]
            row2 = criteria_matrix_normalized[j]
            if i != j and np.all(row2 >= row1) and np.any(row2 > row1):
                dominated = True
                break
        if not dominated:
            non_dominated_indices.append(i)
    return np.array(non_dominated_indices)


def __compute_ideal_point(
    criteria_matrix: np.ndarray,
    non_dominated_indices: np.ndarray,
    gain_directions: np.ndarray,
) -> np.ndarray:
    non_dominated_matrix = criteria_matrix[non_dominated_indices]
    ideal_point = (
        np.max(non_dominated_matrix * gain_directions, axis=0) * gain_directions
    )
    return ideal_point


def __default_distance(
    matrix: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    return np.linalg.norm(matrix - vector, axis=1)


def find_best(
    criteria_matrix: np.ndarray,
    gain_directions: np.ndarray,
    distance_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = __default_distance,
) -> int:
    criteria_matrix_normalized = __min_max_normalize(criteria_matrix)
    non_dominated_indices = __find_non_dominated_rows(
        criteria_matrix_normalized,
        gain_directions,
    )
    ideal_point = __compute_ideal_point(
        criteria_matrix_normalized,
        non_dominated_indices,
        gain_directions,
    )
    distances = distance_func(
        criteria_matrix_normalized[non_dominated_indices],
        ideal_point,
    )
    best_index = non_dominated_indices[np.argmin(distances)]
    return best_index
