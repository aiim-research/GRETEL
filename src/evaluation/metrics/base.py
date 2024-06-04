import numpy as np

from src.core.configurable import Configurable
from abc import ABCMeta, abstractmethod
from src.utils.context import Context
from src.explanation.base import Explanation

class EvaluationMetric(Configurable, metaclass=ABCMeta):

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @abstractmethod
    def evaluate(self, explanation : Explanation):
        pass

    def aggregate(self, measure_list, instances_correctness_list=None):
        return np.mean(measure_list),np.std(measure_list)