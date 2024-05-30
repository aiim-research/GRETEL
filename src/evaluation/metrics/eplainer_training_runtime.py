import time

from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer
from src.explanation.base import Explanation


class ExplainerTrainingRuntimeMetric(EvaluationMetric):

    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger

    def init(self):
        self.name = 'explainer_training_runtime'
        super.__init__()

    def evaluate(self, explanation: Explanation):
        return explanation.explainer_training_runtime