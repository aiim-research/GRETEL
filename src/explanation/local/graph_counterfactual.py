from typing import List

from src.utils.context import Context
from src.core.configurable import Configurable
from src.utils.cfg_utils import retake_dataset, retake_oracle
from src.explanation.local.counterfactual import LocalCounterfactualExplanation
from src.dataset.instances.base import DataInstance
from src.dataset.instances.graph import GraphInstance
from src.utils.context import Context
from src.dataset.dataset_base import Dataset
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer


class LocalGraphCounterfactualExplanation(LocalCounterfactualExplanation):
    """The common logic shared between all Instance-level Graph Counterfactual Explanations should be in this class"""
    
    def __init__(self, 
                 context: Context, 
                 dataset: Dataset, 
                 oracle: Oracle, 
                 explainer: Explainer, 
                 input_instance: GraphInstance, 
                 counterfactual_instances: List[GraphInstance]) -> None:
        # Initializing the base class
        super().__init__(context=context, 
                         dataset=dataset, 
                         oracle=oracle, 
                         explainer=explainer, 
                         input_instance=input_instance, 
                         counterfactual_instances=counterfactual_instances)