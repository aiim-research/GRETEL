from src.evaluation.future.stages.subex_metric_stage import SubexMetricStage
from src.future.explanation.base import Explanation


class SubexRuntime(SubexMetricStage):

    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger


    def init(self):
        super().init()


    def process(self, explanation: Explanation) -> Explanation:
        # Reading the runtime for the target subexplanation
        subex = explanation._info[self.target]
        metric_value = subex._info['runtime']

        # Writing the runtime info in the explanation
        self.write_into_explanation(explanation, metric_value)
  
        # returning the explanation
        return explanation
    