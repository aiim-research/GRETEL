from src.future.explanation.base import Explanation
from src.evaluation.future.stages.subex_metric_stage import SubexMetricStage
from src.utils.metrics.fidelity import fidelity_metric_with_predictions


class SubexFidelity(SubexMetricStage):
    """Similar to correctness measures if the algorithm is producing proper counterfactuals. However, Fidelity measures how faithful they are to the original problem,
       not just to the problem learned by the oracle. It requires a ground truth to be present in the dataset
    """

    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger


    def init(self):
        super().init()


    def process(self, explanation: Explanation) -> Explanation:
        # Reading the oracle calls for the target subexplanation
        subex = explanation._info[self.target]

        oracle = subex.oracle
        input_instance = subex.input_instance
        lbl_input_instance = oracle.predict(input_instance)
        oracle._call_counter -= 1
        aggregated_fidelity = 0
        num_instances = len(subex.counterfactual_instances)

        for cf in subex.counterfactual_instances:
            label_cf = oracle.predict(cf)
            oracle._call_counter -= 1
            cf_fidelity = fidelity_metric_with_predictions(input_instance, lbl_input_instance, label_cf)
            aggregated_fidelity += cf_fidelity

        # Calculating the average fidelity value
        fidelity_value = aggregated_fidelity / num_instances
        # Writing the metric value into the explanation and returning the explanation
        self.write_into_explanation(explanation, fidelity_value)
        return explanation
