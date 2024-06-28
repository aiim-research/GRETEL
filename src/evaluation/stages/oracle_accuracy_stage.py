from src.explanation.base import Explanation
from src.evaluation.stages.metric_stage import MetricStage


class OracleAccuracyStage(MetricStage):
    """
    Meassures the accuracy of the oracle predictions compared to the ground truth labels in the dataset
    """

    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger


    def init(self):
        super().init()


    def process(self, explanation: Explanation) -> Explanation:
        # Unpacking the input instance from te explanation
        instance = explanation.input_instance
        oracle = explanation.oracle

        # Predict the label of the input instance
        predicted_label = oracle.predict(instance)
        oracle._call_counter -= 1
        real_label = instance.label

        # If the predicted label is the same as the ground truth then return 1 and 0 otherwise
        accuracy_value = 1 if (predicted_label == real_label) else 0

        # Writing the metric value into the explanation and returning the explanation
        self.write_into_explanation(explanation, accuracy_value)
        return explanation

        