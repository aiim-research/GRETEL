import os
import pickle
import time
from abc import ABC

import jsonpickle
from scipy import rand

from src.dataset.dataset_base import DataInstance
from src.dataset.dataset_base import Dataset
from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.explainer_base import Explainer
from src.core.oracle_base import Oracle

class ActionabilityEvaluator(ABC):

    def __init__(self, id):
        super().__init__()
        self.id = id
        self.restricted_edges = []
        self.restricted_nodes = []
        self.restricted_node_features = []
        self.restricted_edge_features

    def evaluate_graph(original_instance : DataInstance, 
                       cf_instance : DataInstance):
        return True

    def evaluate_edge(original_instance : DataInstance, 
                      cf_instance : DataInstance, 
                      target_edge):
        return True

    def evaluate_node(original_instance : DataInstance, 
                      cf_instance : DataInstance, 
                      target_node):
        return True

    def evaluate_node_features(original_instance : DataInstance, 
                               original_features, 
                               new_features):
        return True