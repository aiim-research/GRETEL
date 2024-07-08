import numpy as np

from src.evaluation.metrics.base import EvaluationMetric
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.utils.metrics.sparsity import sparsity_metric


class SparsityMetric(EvaluationMetric):
    """Provides the ratio between the number of features modified to obtain the counterfactual example
     and the number of features in the original instance. Only considers structural features.
    """

    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger

    def init(self):
        super().init()
        self.name = 'sparsity'

    def evaluate(self, explanation: LocalGraphCounterfactualExplanation):
        # Get the input instance from the explanation and get its label
        input_inst = explanation.input_instance
        aggregated_sparsity = 0
        correct_instances = 0
        for cf in explanation.counterfactual_instances:
            sparsity = sparsity_metric(input_inst, cf)
            if sparsity > 0:
                aggregated_sparsity += sparsity
                correct_instances += 1
        if correct_instances == 0:
            return 0
        return aggregated_sparsity / correct_instances

    @classmethod
    def aggregate(cls, measure_list, instances_correctness_list=None):
        # If no correctness list is provided aggregate all the measures
        if instances_correctness_list is None:
            return super().aggregate(measure_list, instances_correctness_list)
        else: # If correctness list is provided then aggregate only the measures of the correct instances
            filtered_measure_list = [item for item, flag in zip(measure_list, instances_correctness_list) if flag > 0.0]

            # Avoid aggregating an empty list
            if len(filtered_measure_list) > 0:
                return np.mean(filtered_measure_list), np.std(filtered_measure_list)
            else:
                return 0.0, 0.0
