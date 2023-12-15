import copy
import sys
from abc import ABC

from src.core.explainer_base import Explainer
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric
from src.evaluation.evaluation_metric_correctness import CorrectnessMetric
from src.core.configurable import Configurable
import numpy as np

from src.utils.cfg_utils import get_dflts_to_of, init_dflts_to_of, inject_dataset, inject_oracle, retake_oracle, retake_dataset

class ExplanationAggregator(Configurable):

    def init(self):
        super().init()
        self.dataset = retake_dataset(self.local_config)
        self.oracle = retake_oracle(self.local_config)

    def aggregate(self, org_instance, explanations):
        pass