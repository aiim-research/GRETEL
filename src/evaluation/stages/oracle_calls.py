from src.future.explanation.base import Explanation
from src.evaluation.stages.metric_stage import MetricStage


class OracleCalls(MetricStage):
    """Provides the number of calls to the oracle an explainer has to perform in order to generate
       a counterfactual example
    """

    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger


    def init(self):
        super().init()

    def process(self, explanation: Explanation) -> Explanation:
        # Getting the number of oracle calls from the Oracle object
        oracle = explanation.oracle
        oracle_calls_value = oracle.get_calls_count()
        oracle.reset_call_count()

        # Writing the metric value into the explanation and returning the explanation
        self.write_into_explanation(explanation, oracle_calls_value)
        return explanation

        