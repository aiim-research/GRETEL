from src.explanation.base import Explanation
from src.evaluation.stages.metric_stage import MetricStage


class Correctness(MetricStage):
    """
    Verifies that the class from the counterfactual example is different from that of the original instance
    """

    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger


    def init(self):
        super().init()


    def process(self, explanation: Explanation) -> Explanation:
        input_inst_lbl = explanation.oracle.predict(explanation.input_instance)
        explanation.oracle._call_counter -= 1

        correctness = 0
        for cf in explanation.counterfactual_instances:
            # Checking if the counterfactual instances are correct
            if explanation.oracle.predict(cf) != input_inst_lbl:
                correctness += 1
            explanation.oracle._call_counter -= 1

        correctness /= len(explanation.counterfactual_instances)

        self.write_into_explanation(explanation, correctness)

        return explanation

        