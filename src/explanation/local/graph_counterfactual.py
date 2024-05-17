from abc import ABCMeta, abstractmethod
from typing import List

from src.utils.context import Context
from src.core.configurable import Configurable
from src.utils.cfg_utils import retake_dataset, retake_oracle
from src.explanation.local.counterfactual import LocalCounterfactualExplanation
from src.dataset.instances.base import DataInstance
from src.dataset.instances.graph import GraphInstance


class LocalGraphCounterfactualExplanation(LocalCounterfactualExplanation):
    """The common logic shared between all Instance-level Graph Counterfactual Explanations should be in this class"""
    
    def __init__(self, explainer_class: str, input_instance: GraphInstance, counterfactual_instances: List[GraphInstance]) -> None:
        super().__init__(explainer_class=explainer_class, input_instance=input_instance, counterfactual_instances=counterfactual_instances)