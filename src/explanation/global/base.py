from abc import ABCMeta, abstractmethod
from typing import List

from src.utils.context import Context
from src.core.configurable import Configurable
from src.utils.cfg_utils import retake_dataset, retake_oracle
from src.explanation.base import Explanation
from src.dataset.instances.base import DataInstance


class GlobalExplanation(Explanation):
    """The common logic shared between all instance-level explanation types should be in this class"""
    
    def __init__(self, explainer_class, input_instances: List[DataInstance]) -> None:
        super().__init__(explainer_class=explainer_class)
        self._input_instances = input_instances

    @property
    def input_instances(self) -> List[DataInstance]:
        return self._input_instances
    
    @input_instances.setter
    def input_instances(self, new_input_instances):
        self.input_instances = new_input_instances