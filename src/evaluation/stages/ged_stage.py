import numpy as np

from src.utils.metrics.ged import GraphEditDistanceMetric as ged
from src.explanation.base import Explanation
from src.evaluation.stages.metric_stage import MetricStage


class GraphEditDistanceStage(MetricStage):
    """Provides a graph edit distance function for graphs where nodes are already matched, 
    thus eliminating the need of performing an NP-Complete graph matching.
    """

    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger

        if 'node_add_cost' not in self.local_config['parameters']:
            self.local_config['parameters']['node_add_cost'] = 1.0
        
        if 'node_rem_cost' not in self.local_config['parameters']:
            self.local_config['parameters']['node_rem_cost'] = 1.0

        if 'edge_add_cost' not in self.local_config['parameters']:
            self.local_config['parameters']['edge_add_cost'] = 1.0
        
        if 'edge_rem_cost' not in self.local_config['parameters']:
            self.local_config['parameters']['edge_rem_cost'] = 1.0


    def init(self):
        super().init()

        node_add_cost = self.local_config['parameters']['node_add_cost']
        node_rem_cost = self.local_config['parameters']['node_rem_cost']
        edge_add_cost = self.local_config['parameters']['edge_add_cost']
        edge_rem_cost = self.local_config['parameters']['edge_rem_cost']

        self.dst = ged(node_insertion_cost=node_add_cost, 
                       node_deletion_cost=node_rem_cost,
                       edge_insertion_cost=edge_add_cost,
                       edge_deletion_cost=edge_rem_cost,
                       undirected=True)


    def process(self, explanation: Explanation) -> Explanation:
        # Get the input instance from the explanation and get its label
        input_inst = explanation.input_instance
        input_inst_lbl = explanation.oracle.predict(input_inst)
        explanation.oracle._call_counter -= 1

        # Set in the edit distance function if the graph is directed or not based on the input instance
        self.dst.undirected = not input_inst.directed

        # Iterate over the counterfactual examples and calculate the mean ged of the explanation
        aggregated_ged = 0.0
        correct_instances = 0.0
        for cf in explanation.counterfactual_instances:
            cf_ged = self.dst.evaluate(instance_1=input_inst, 
                                       instance_2=cf, 
                                       oracle=explanation.oracle, 
                                       explainer=explanation.explainer, 
                                       dataset=explanation.dataset)
         
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

        