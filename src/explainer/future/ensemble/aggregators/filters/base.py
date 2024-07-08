from typing import List

from src.core.oracle_base import Oracle
from src.dataset.dataset_base import Dataset
from src.core.configurable import Configurable
from src.utils.cfg_utils import retake_oracle, retake_dataset
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation

class ExplanationFilter(Configurable):

    def init(self):
        self.dataset: Dataset = retake_dataset(self.local_config)
        self.oracle: Oracle = retake_oracle(self.local_config)

        super().init()


    def filter(self, explanations: List[LocalGraphCounterfactualExplanation]) -> List[LocalGraphCounterfactualExplanation]:
        if len(explanations) < 1:
            raise ValueError('The list of explanations to be filtered cannot be empty')
        
        return self.real_filter(explanations)
    
    
    def real_filter(self, explanations: List[LocalGraphCounterfactualExplanation]) -> List[LocalGraphCounterfactualExplanation]:
        """
        This method should be reimplemented by the child classes. The default behavior is to return the input explanations list.
        This is useful in case a filter is not needed.
        """
        return explanations