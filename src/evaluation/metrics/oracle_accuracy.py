from src.evaluation.metrics.base import EvaluationMetric
from src.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.utils.metrics.ged import GraphEditDistanceMetric


class OracleAccuracyMetric(EvaluationMetric):
    """As correctness measures if the algorithm is producing counterfactuals, but in Fidelity measures how faithful they are to the original problem,
     not just to the problem learned by the oracle. Requires a ground truth in the dataset
    """

    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger


    def init(self):
        super().init()
        self.name = 'oracle_accuracy'


    def evaluate(self, explanation: LocalGraphCounterfactualExplanation):
        # Unpacking the input instance from te explanation
        instance = explanation.input_instance
        oracle = explanation.oracle

        # Predict the label of the input instance
        predicted_label_instance_1 = oracle.predict(instance)
        oracle._call_counter -= 1
        real_label_instance_1 = instance.label

        # If the predicted label is the same as the ground truth then return 1 and 0 otherwise
        result = 1 if (predicted_label_instance_1 == real_label_instance_1) else 0
        
        return result