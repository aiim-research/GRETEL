from copy import deepcopy
from typing import List
import numpy as np
import sys
from abc import ABCMeta, abstractmethod

from src.core.factory_base import get_instance_kvargs
from src.core.oracle_base import Oracle
from src.dataset.dataset_base import Dataset
from src.core.configurable import Configurable
from src.core.trainable_base import Trainable
from src.dataset.instances.base import DataInstance
from src.dataset.instances.graph import GraphInstance
from src.explainer.future.ensemble.aggregators.nodes.base import NodeFeatureAggregator
from src.utils.cfg_utils import init_dflts_to_of, inject_dataset, inject_oracle, retake_oracle, retake_dataset
from src.utils.utils import pad_adj_matrix
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.explainer.future.ensemble.aggregators.filters.base import ExplanationFilter


class ExplanationMinimizer(Trainable):

    def check_configuration(self):
        super().check_configuration()


    def init(self):
        self.logger = self.context.logger
        self.dataset: Dataset = retake_dataset(self.local_config)
        self.oracle: Oracle = retake_oracle(self.local_config)

        super().init()


    def real_fit(self):
        pass
    
    @abstractmethod
    def minimize(self, instance, explaination: LocalGraphCounterfactualExplanation) -> DataInstance:
        pass
    
    
    def get_edge_differences(self, instance: DataInstance, cf_instance: DataInstance):
        # Summing the two adjacency matrices (the metrices need to have the same size) edges that appear only in one of the two instances are the different ones
        edge_freq_matrix = np.add(instance.data, cf_instance.data)
        diff_matrix = np.where(edge_freq_matrix == 1, 1, 0)
        diff_number = np.count_nonzero(diff_matrix)

        if instance.directed:
            filtered_diff_number = int(diff_number)
        else:
            filtered_diff_number = int(diff_number/2)

        return filtered_diff_number, diff_matrix