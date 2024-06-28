import numpy as np

from src.explainer.ensemble.aggregators.multi_criteria.distances.base_distance import (
    BaseDistance,
)


class ChebyshevDistance(BaseDistance):
    def calculate(
        self,
        matrix: np.ndarray,
        vector: np.ndarray,
    ) -> np.ndarray:
        return np.max(np.abs(matrix - vector), axis=1)
