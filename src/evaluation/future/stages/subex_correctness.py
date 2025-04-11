from src.future.explanation.base import Explanation
from src.evaluation.future.stages.metric_stage import MetricStage
from src.evaluation.future.stages.subex_metric_stage import SubexMetricStage


class Correctness(SubexMetricStage):
    """
    Verifies that the class from the counterfactual example is different from that of the original instance
    """

    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger


    def init(self):
        super().init()


    def process(self, explanation: Explanation) -> Explanation:
        # Get the subexplanation from the explanation
        subex = explanation._info[self.target]

        input_inst_lbl = subex.oracle.predict(subex.input_instance)
        subex.oracle._call_counter -= 1

        correctness = 0
        for cf in subex.counterfactual_instances:
            # Checking if the counterfactual instances are correct
            if subex.oracle.predict(cf) != input_inst_lbl:
                correctness += 1
            subex.oracle._call_counter -= 1

        correctness /= len(subex.counterfactual_instances)

        self.write_into_explanation(explanation, correctness)

        return explanation

        