import numpy as np

from src.explanation.base import Explanation
from src.evaluation.stages.metric_stage import MetricStage
from src.utils.metrics.ged import GraphEditDistanceMetric


class SparsityStage(MetricStage):
    """Provides the ratio between the number of features modified to obtain the counterfactual example
       and the number of features in the original instance. Only considers structural features.
    """

    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger


    def init(self):
        super().init()
        self.dst = GraphEditDistanceMetric()


    def process(self, explanation: Explanation) -> Explanation:
        # Get the input instance from the explanation and get its label
        input_inst = explanation.input_instance

        aggregated_sparsity = 0
        correct_instances = 0

        self.dst.undirected = not input_inst.directed
        for cf in explanation.counterfactual_instances:
           cf_ged = self.dst.evaluate(input_inst, cf, explanation.oracle, explanation.explainer, explanation.dataset)

           if cf_ged > 0:
            cf_nodes = cf.data.shape[0]
            cf_edges = np.count_nonzero(cf.data)
            if not cf.directed:
                cf_edges /= 2

            cf_struct_features = cf_nodes +cf_edges

            aggregated_sparsity += cf_ged/cf_struct_features
            correct_instances += 1

        # Aggregating the sparsity, considering only correct counterfactual instances
        sparsity_metric = 0.0
        if correct_instances > 0:
            sparsity_metric = aggregated_sparsity / correct_instances

        # Saving the sparsity value in the explanation and returning the explanation
        self.write_into_explanation(explanation, sparsity_metric)
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

        