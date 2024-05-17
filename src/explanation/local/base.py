from abc import ABCMeta, abstractmethod

from src.utils.context import Context
from src.core.configurable import Configurable
from src.utils.cfg_utils import retake_dataset, retake_oracle
from src.explanation.base import Explanation
from src.dataset.instances.base import DataInstance


class LocalExplanation(Explanation):
    """The common logic shared between all instance-level explanation types should be in this class"""
    
    def __init__(self, explainer_class: str, input_instance: DataInstance) -> None:
        super().__init__(explainer_class=explainer_class)
        self._id = input_instance.id
        self._input_instance = input_instance
        

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, new_id) -> None:
        self._id = new_id

    @property
    def input_instance(self) -> DataInstance:
        return self._input_instance
    
    @input_instance.setter
    def input_instance(self, new_input_instance) -> None:
        self.input_instance = new_input_instance