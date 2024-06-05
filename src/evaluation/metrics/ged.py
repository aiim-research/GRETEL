import numpy as np

from src.evaluation.metrics.base import EvaluationMetric
from src.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.utils.metrics.ged import GraphEditDistanceMetric as ged


class GraphEditDistanceMetric(EvaluationMetric):
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
        self.name = 'graph_edit_distance'

        node_add_cost = self.local_config['parameters']['node_add_cost']
        node_rem_cost = self.local_config['parameters']['node_rem_cost']
        edge_add_cost = self.local_config['parameters']['edge_add_cost']
        edge_rem_cost = self.local_config['parameters']['edge_rem_cost']

        self.dst = ged(node_insertion_cost=node_add_cost, 
                       node_deletion_cost=node_rem_cost,
                       edge_insertion_cost=edge_add_cost,
                       edge_deletion_cost=edge_rem_cost,
                       undirected=True)
        

    def evaluate(self, explanation: LocalGraphCounterfactualExplanation):
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

        if correct_instances > 0:
            return aggregated_ged/correct_instances
        else:
            return 0.0
            
    
    def aggregate(self, measure_list, instances_correctness_list=None):
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

    