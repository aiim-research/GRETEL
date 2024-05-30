from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer
from src.utils.metrics.ged import GraphEditDistanceMetric
from src.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation




class CorrectnessMetric(EvaluationMetric):
    """
    Verifies that the class from the counterfactual example is different from that of the original instance
    """

    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger


    def init(self):
        self.name = 'correctness'
        super.__init__()


    def evaluate(self, explanation: LocalGraphCounterfactualExplanation):
        input_inst_lbl = explanation.oracle.predict(explanation.input_instance)
        explanation.oracle._call_counter -= 1

        correctness = 0
        for cf in explanation.counterfactual_instances:
            if explanation.oracle.predict(cf) != input_inst_lbl:
                correctness += 1
            explanation.oracle._call_counter -= 1

        correctness /= len(explanation.counterfactual_instances)

        return correctness