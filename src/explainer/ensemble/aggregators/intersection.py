import copy
import sys
from abc import ABC

from src.core.explainer_base import Explainer
from src.dataset.instances.graph import GraphInstance
from src.explainer.ensemble.aggregators.base import ExplanationAggregator
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric
import numpy as np

from src.core.factory_base import get_instance_kvargs
from src.utils.cfg_utils import get_dflts_to_of, init_dflts_to_of, inject_dataset, inject_oracle, retake_oracle, retake_dataset
from src.utils.utils import pad_adj_matrix


class ExplanationIntersection(ExplanationAggregator):

    def real_aggregate(self, instance, explanations):
        
        label = self.oracle.predict(instance)

        max_dim = max([exp.data.shape[0] for exp in explanations])
        edge_freq_matrix = np.ones((max_dim, max_dim))

        for exp in explanations:
            if self.oracle.predict(exp) != label:
                edge_freq_matrix = edge_freq_matrix * pad_adj_matrix(exp.data, max_dim)

        adj = np.where(edge_freq_matrix > 0, 1, 0)

        return GraphInstance(id=instance.id, data=adj, label=1-instance.label)
        