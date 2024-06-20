from abc import ABCMeta, abstractmethod

import numpy as np

from src.core.configurable import Configurable


class BaseDistance(Configurable, metaclass=ABCMeta):
    @abstractmethod
    def calculate(
        self,
        matrix: np.ndarray,
        vector: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError
