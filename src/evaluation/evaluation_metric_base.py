import numpy as np
from src.core.explainer_base import Explainer
from src.core.oracle_base import Oracle

import networkx as nx

from abc import ABC, abstractmethod

class EvaluationMetric(ABC):

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
    def evaluate(self, instance_1 , instance_2 , oracle : Oracle=None, explainer : Explainer=None, dataset  = None):
        pass

    def aggregate(self,measure_list):
        return np.mean(measure_list),np.std(measure_list)