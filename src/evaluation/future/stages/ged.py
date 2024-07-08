import numpy as np

from src.utils.metrics.ged import graph_edit_distance_metric
from src.future.explanation.base import Explanation
from src.evaluation.future.stages.metric_stage import MetricStage


class GraphEditDistance(MetricStage):
    """Provides a graph edit distance function for graphs where nodes are already matched, 
    thus eliminating the need of performing an NP-Complete graph matching.
    """

    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger

    def init(self):
        super().init()

    def process(self, explanation: Explanation) -> Explanation:
        # Get the input instance from the explanation and get its label
        input_inst = explanation.input_instance
        input_inst_lbl = explanation.oracle.predict(input_inst)
        explanation.oracle._call_counter -= 1

        # Iterate over the counterfactual examples and calculate the mean ged of the explanation
        aggregated_ged = 0.0
        correct_instances = 0.0
        for cf in explanation.counterfactual_instances:
            cf_ged = graph_edit_distance_metric(input_inst.data, cf.data, input_inst.directed and cf.directed)

            cf_lbl = explanation.oracle.predict(cf)
            explanation.oracle._call_counter -= 1

            # Not consider in the average the ged of incorrect instances
            if input_inst_lbl != cf_lbl:
                correct_instances += 1
                aggregated_ged += cf_ged

        # Calculating the average GED considering only the correct instances
        ged_metric = 0.0
        if correct_instances > 0:
            ged_metric = aggregated_ged/correct_instances
        
        # Writing the metric value into the explanation and returning the explanation
        self.write_into_explanation(explanation, ged_metric)
        return explanation

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
