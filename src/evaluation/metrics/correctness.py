from src.evaluation.metrics.base import EvaluationMetric
from src.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation




class CorrectnessMetric(EvaluationMetric):
    """
    Verifies that the class from the counterfactual example is different from that of the original instance
    """

    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger


    def init(self):
        super().init()
        self.name = 'correctness'


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