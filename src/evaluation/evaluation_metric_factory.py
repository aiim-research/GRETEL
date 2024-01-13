from src.evaluation.evaluation_metric_fidelity_node import FidelityNodeMetric
from src.evaluation.evaluation_metric_oracle_accuracy_node_classification import OracleAccuracyNodeMetric
from src.evaluation.evaluation_metric_correctness import CorrectnessMetric
from src.evaluation.evaluation_metric_fidelity import FidelityMetric
from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric
from src.evaluation.evaluation_metric_runtime import RuntimeMetric
from src.evaluation.evaluation_metric_sparsity import SparsityMetric
from src.evaluation.evaluation_metric_oracle_calls import OracleCallsMetric
from src.evaluation.evaluation_metric_oracle_accuracy import OracleAccuracyMetric
from src.evaluation.evaluation_metric_smiles_levenshtein import SmilesLevenshteinMetric
from src.evaluation.evaluation_metric_dumper import InstancesDumper



class EvaluationMetricFactory:

    def __init__(self,config_dict) -> None:
        self._config_dict = config_dict
        self._evaluation_metric_id_counter = 0

    def get_evaluation_metric_by_name(self, metric_dict) -> EvaluationMetric:
        metric_name = metric_dict['name']
        metric_parameters = metric_dict['parameters']

        if(metric_name == 'graph_edit_distance'):
            return self.get_graph_edit_distance_metric(config_dict=metric_dict)

        elif metric_name == 'oracle_calls':
            return self.get_oracle_calls_metric(config_dict=metric_dict)

        elif metric_name == 'sparsity':
            return self.get_sparsity_metric(config_dict=metric_dict)

        elif metric_name == 'correctness':
            return self.get_correctness_metric(config_dict=metric_dict)

        elif metric_name == 'fidelity':
            return self.get_fidelity_metric(config_dict=metric_dict)
        
        elif metric_name == 'fidelity_node':
            return self.get_fidelity_node_metric(config_dict=metric_dict)

        elif metric_name == 'oracle_accuracy':
            return self.get_oracle_accuracy_metric(config_dict=metric_dict)

        elif metric_name == 'smiles_levenshtein':
            return self.get_smiles_levenshtein_metric(config_dict=metric_dict)
        
        elif metric_name == 'oracle_accuracy_node':
            return self.get_oracle_accuracy_node_metric(config_dict=metric_dict)
        
        elif metric_name == 'dumper':
            return self.get_dumper_metric(config_dict=metric_dict)
        
        elif metric_name == 'runtime':
            return self.get_runtime_metric(config_dict=metric_dict)

        else:
            raise ValueError('''The provided evaluation metric name does not match any evaluation
             metric provided by the factory''')
        
    def get_runtime_metric(self, config_dict=None) -> EvaluationMetric:
        result = RuntimeMetric(config_dict)
        return result

    def get_dumper_metric(self, config_dict=None) -> EvaluationMetric:
        result = InstancesDumper(self._config_dict,config_dict)
        return result

    def get_correctness_metric(self, config_dict=None) -> EvaluationMetric:
        result = CorrectnessMetric(config_dict)
        return result

    def get_oracle_calls_metric(self, config_dict=None) -> EvaluationMetric:
        result = OracleCallsMetric(config_dict)
        return result

    def get_graph_edit_distance_metric(self, node_insertion_cost=1.0, node_deletion_cost=1.0, 
                                        edge_insertion_cost=1.0, edge_deletion_cost=1.0, undirected=True, config_dict=None) -> EvaluationMetric:
        
        result = GraphEditDistanceMetric(node_insertion_cost, node_deletion_cost, edge_insertion_cost, 
                                            edge_deletion_cost, undirected, config_dict)

        return result


    def get_sparsity_metric(self, config_dict=None) -> EvaluationMetric:
        result = SparsityMetric(config_dict)
        return result


    def get_fidelity_metric(self, config_dict=None) -> EvaluationMetric:
        result = FidelityMetric(config_dict)
        return result

    
    def get_fidelity_node_metric(self, config_dict=None) -> EvaluationMetric:
        result = FidelityNodeMetric(config_dict)
        return result

    def get_oracle_accuracy_metric(self, config_dict=None) -> EvaluationMetric:
        result = OracleAccuracyMetric(config_dict)
        return result

    def get_oracle_accuracy_node_metric(self, config_dict=None) -> EvaluationMetric:
        result = OracleAccuracyNodeMetric(config_dict)
        return result
    
    def get_smiles_levenshtein_metric(self, config_dict=None) -> EvaluationMetric:
        result = SmilesLevenshteinMetric(config_dict)
        return result

