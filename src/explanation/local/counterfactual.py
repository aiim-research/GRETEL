from typing import List

from src.explanation.local.base import LocalExplanation
from src.dataset.instances.base import DataInstance
from src.utils.context import Context
from src.dataset.dataset_base import Dataset
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer


class LocalCounterfactualExplanation(LocalExplanation):
    """The common logic shared between all instance-level counterfactual explanations should be in this class"""
    
    def __init__(self, 
                 context: Context, 
                 dataset: Dataset, 
                 oracle: Oracle, 
                 explainer: Explainer, 
                 input_instance: DataInstance, 
                 counterfactual_instances: List[DataInstance]) -> None:
        super().__init__(context=context, dataset=dataset, oracle=oracle, explainer=explainer, input_instance=input_instance)
        
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