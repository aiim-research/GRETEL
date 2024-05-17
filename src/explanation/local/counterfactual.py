from abc import ABCMeta, abstractmethod
from typing import List

from src.utils.context import Context
from src.core.configurable import Configurable
from src.utils.cfg_utils import retake_dataset, retake_oracle
from src.explanation.local.base import LocalExplanation
from src.dataset.instances.base import DataInstance


class LocalCounterfactualExplanation(LocalExplanation):
    """The common logic shared between all instance-level counterfactual explanations should be in this class"""
    
    def __init__(self, explainer_class: str, input_instance: DataInstance, counterfactual_instances: List[DataInstance]) -> None:
        super().__init__(explainer_class=explainer_class, input_instance=input_instance)

        if len(counterfactual_instances) < 1:
            raise ValueError('The number of explanations instances in an explanation should be grater than 0')
        
        self._counterfactual_instances = counterfactual_instances

    
    @property
    def counterfactual_instances(self) -> List[DataInstance]:
        return self._counterfactual_instances
    
    @counterfactual_instances.setter
    def counterfactual_instances(self, new_counterfactual_instances) -> None:
        self._counterfactual_instances = new_counterfactual_instances

    @property
    def top(self) -> DataInstance:
        return self._counterfactual_instances[0]