from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer


class OracleCallsMetric(EvaluationMetric):
    """Provides the number of calls to the oracle an explainer has to perform in order to generate
    a counterfactual example
    """

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Oracle_Calls'

    def evaluate(self, instance_1 , instance_2 , oracle : Oracle=None, explainer : Explainer=None, dataset = None):
        result = oracle.get_calls_count()
        oracle.reset_call_count()
        return result