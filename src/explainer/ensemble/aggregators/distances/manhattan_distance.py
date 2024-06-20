import numpy as np

from src.explainer.ensemble.aggregators.distances.base_distance import BaseDistance


class ManhattanDistance(BaseDistance):
    def calculate(
        self,
        matrix: np.ndarray,
        vector: np.ndarray,
    ) -> np.ndarray:
        return np.sum(np.abs(matrix - vector), axis=1)
