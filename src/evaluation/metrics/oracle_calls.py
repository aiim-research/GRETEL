from src.evaluation.metrics.base import EvaluationMetric
from src.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation


class OracleCallsMetric(EvaluationMetric):
    """Provides the number of calls to the oracle an explainer has to perform in order to generate
    a counterfactual example
    """

    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger


    def init(self):
        super().init()
        self.name = 'oracle_calls'


    def evaluate(self, explanation: LocalGraphCounterfactualExplanation):
        oracle = explanation.oracle
        result = oracle.get_calls_count()
        oracle.reset_call_count()
        return result