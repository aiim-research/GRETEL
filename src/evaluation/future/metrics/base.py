import numpy as np
from abc import ABCMeta, abstractmethod

from src.core.configurable import Configurable
from src.utils.context import Context
from src.future.explanation.base import Explanation

class EvaluationMetric(Configurable, metaclass=ABCMeta):

    @abstractmethod
    def check_configuration(self):
        pass

    @abstractmethod
    def init(self):
        pass

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @abstractmethod
    def evaluate(self, explanation : Explanation):
        pass

    @classmethod
    def aggregate(cls, measure_list, instances_correctness_list=None):
        return np.mean(measure_list),np.std(measure_list)