from src.evaluation.metrics.base import EvaluationMetric
from src.explanation.base import Explanation


class RuntimeMetric(EvaluationMetric):

    def check_configuration(self):
        super().check_configuration()
        self.logger = self.context.logger


    def init(self):
        super().init()
        self.name = 'explanation_runtime'
        

    def evaluate(self, explanation: Explanation):
        return explanation.explanation_runtime