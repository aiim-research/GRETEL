from abc import ABCMeta, abstractmethod
from typing import List

from src.utils.context import Context
from src.core.configurable import Configurable
from src.utils.cfg_utils import retake_dataset, retake_oracle
from src.explanation.base import Explanation
from src.dataset.instances.base import DataInstance
from src.utils.context import Context
from src.dataset.dataset_base import Dataset
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer


class GlobalExplanation(Explanation):
    """The common logic shared between all instance-level explanation types should be in this class"""
    
    def __init__(self, context: Context, dataset: Dataset, oracle: Oracle, explainer: Explainer, input_instances: List[DataInstance]) -> None:
        super().__init__(context=context, dataset=dataset, oracle=oracle, explainer=explainer)
        self._input_instances = input_instances

    @property
    def input_instances(self) -> List[DataInstance]:
        return self._input_instances
    
    @input_instances.setter
    def input_instances(self, new_input_instances):
        self.input_instances = new_input_instances