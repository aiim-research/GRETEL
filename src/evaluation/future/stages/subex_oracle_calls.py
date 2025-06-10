from src.future.explanation.base import Explanation
from src.evaluation.future.stages.subex_metric_stage import SubexMetricStage


class SubexOracleCalls(SubexMetricStage):
    """Provides the number of calls to the oracle an explainer has to perform in order to generate
       a counterfactual example
    """

    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger


    def init(self):
        super().init()

    def process(self, explanation: Explanation) -> Explanation:
        # Reading the oracle calls for the target subexplanation
        subex = explanation._info[self.target]
        metric_value = subex._info['oracle_calls']

        # Writing the metric value into the explanation and returning the explanation
        self.write_into_explanation(explanation, metric_value)
        return explanation

        