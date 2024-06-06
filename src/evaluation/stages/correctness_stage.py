from src.explanation.base import Explanation
from src.evaluation.stages.measurement_stage import MeasurementStage


class CorrectnessStage(MeasurementStage):
    """
    Verifies that the class from the counterfactual example is different from that of the original instance
    """

    def process(self, explanation: Explanation) -> Explanation:
        input_inst_lbl = explanation.oracle.predict(explanation.input_instance)
        explanation.oracle._call_counter -= 1

        correctness = 0
        for cf in explanation.counterfactual_instances:
            if explanation.oracle.predict(cf) != input_inst_lbl:
                correctness += 1
            explanation.oracle._call_counter -= 1

        correctness /= len(explanation.counterfactual_instances)

        explanation._stages_info[self.__class__.name] = correctness

        return explanation

        