import copy
import sys
import numpy as np
import random
from typing import List

from src.explainer.future.meta.minimizer.base import ExplanationMinimizer
from src.dataset.instances.base import DataInstance
from src.dataset.instances.graph import GraphInstance
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.utils.comparison import get_all_edge_differences, get_edge_differences
from src.utils.metrics.ged import GraphEditDistanceMetric


class Dummy(ExplanationMinimizer):

    def check_configuration(self):
        super().check_configuration()

    
    def init(self):
        super().init()
        self.distance_metric = GraphEditDistanceMetric()  


    def minimize(self, explaination: LocalGraphCounterfactualExplanation) -> DataInstance:
        min_ctf = explaination.counterfactual_instances[0]

        # Return the minimal counterfactual
        return min_ctf

    def write(self):
        pass

    def read(self):
        pass