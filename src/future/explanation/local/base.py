from abc import ABCMeta, abstractmethod

from src.utils.context import Context
from src.core.configurable import Configurable
from src.utils.cfg_utils import retake_dataset, retake_oracle
from src.future.explanation.base import Explanation
from src.dataset.instances.base import DataInstance
from src.utils.context import Context
from src.dataset.dataset_base import Dataset
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer


class LocalExplanation(Explanation):
    """The common logic shared between all instance-level explanation types should be in this class"""
    
    def __init__(self, context: Context, dataset: Dataset, oracle: Oracle, explainer: Explainer, input_instance: DataInstance) -> None:
        super().__init__(context=context, dataset=dataset, oracle=oracle, explainer=explainer)
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