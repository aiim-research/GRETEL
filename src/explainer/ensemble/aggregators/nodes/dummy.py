from typing import List
from copy import deepcopy
import numpy as np

from src.dataset.instances.base import DataInstance
from src.explainer.ensemble.aggregators.nodes.base import NodeFeatureAggregator
from src.utils.utils import pad_features
from src.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.dataset.instances.graph import GraphInstance

class DummyAggregator(NodeFeatureAggregator):

    def aggregate(self, aggregated_explanation: LocalGraphCounterfactualExplanation, base_explanations: List[LocalGraphCounterfactualExplanation]):
        return aggregated_explanation.counterfactual_instances


