from src.evaluation.metrics.base import EvaluationMetric
from src.explanation.base import Explanation


class RuntimeMetric(EvaluationMetric):

    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger

    def init(self):
        self.name = 'explanation_runtime'
        super.__init__()

    def evaluate(self, explanation: Explanation):
        return explanation.explanation_runtime