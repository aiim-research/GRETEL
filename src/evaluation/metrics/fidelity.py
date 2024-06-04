from src.evaluation.metrics.base import EvaluationMetric
from src.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.utils.metrics.ged import GraphEditDistanceMetric


class FidelityMetric(EvaluationMetric):
    """As correctness measures if the algorithm is producing counterfactuals, but in Fidelity measures how faithful they are to the original problem,
     not just to the problem learned by the oracle. Requires a ground truth in the dataset
    """

    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger


    def init(self):
        super.__init__()
        self.name = 'sparsity'
        self.dst = GraphEditDistanceMetric()


    def evaluate(self, explanation: LocalGraphCounterfactualExplanation):
        oracle = explanation.oracle
        input_instance = explanation.input_instance
        lbl_input_instance = oracle.predict(input_instance)
        oracle._call_counter -= 1

        # This term indicates how reliable is the oracle prediction
        prediction_fidelity = 1 if (lbl_input_instance == input_instance.label) else 0

        aggregated_fidelity = 0
        num_instances = len(explanation.counterfactual_instances)

        for cf in explanation.counterfactual_instances:
            label_cf = oracle.predict(cf)
            oracle._call_counter -= 1

            counterfactual_fidelity = 1 if (label_cf == input_instance.label) else 0
            cf_fidelity = prediction_fidelity - counterfactual_fidelity

            aggregated_fidelity += cf_fidelity
        
        return aggregated_fidelity / num_instances