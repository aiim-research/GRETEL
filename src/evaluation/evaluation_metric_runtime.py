import time
from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer


class RuntimeMetric(EvaluationMetric):

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Runtime'
        self._special = True

    def evaluate(self, instance_1 , instance_2 , oracle : Oracle=None, explainer : Explainer=None, dataset = None):
        start_time = time.time()
        counterfactual = explainer.explain(instance_1)
        end_time = time.time()
        # giving the same id to the counterfactual and the original instance 
        counterfactual.id = instance_1.id       
        
        return (end_time - start_time), counterfactual