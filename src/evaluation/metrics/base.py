import numpy as np

from src.core.configurable import Configurable
from abc import ABC, abstractmethod

class EvaluationMetric(Configurable, ABC):

    def __init__(self, config_dict=None) -> None:
        super().__init__()
        self._name = 'abstract_metric'
        self._config_dict = config_dict
        self._special = False #TODO: this must be removed in the future just to manage Runtime NOW QUICKFIX 

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @abstractmethod
    def evaluate(self, explanation):
        pass

    def aggregate(self, measure_list, instances_correctness_list=None):
        return np.mean(measure_list),np.std(measure_list)