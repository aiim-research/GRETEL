from os.path import join
from os import makedirs
from src.dataset.dataset_base import Dataset
from src.dataset.instances.base import DataInstance
from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer
from src.evaluation.evaluation_metric_correctness import CorrectnessMetric

import jsonpickle


class InstancesDumper(EvaluationMetric):

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Dumper'
        self._correctness = CorrectnessMetric()
        self._store_path = join(config_dict['parameters']['store_path'], self.__class__.__name__)
        makedirs(self._store_path, exist_ok=True)

    def evaluate(self, original: DataInstance, counterfactual: DataInstance, oracle: Oracle=None, explainer: Explainer=None, dataset = None):
        output_path = self.__create_dirs(oracle, explainer, dataset)
        results_uri = join(output_path, f'cf_{original.id}.json')
      
        correctness = self._correctness.evaluate(original, counterfactual, oracle)
        
        info = {
            "orginal_id":original.id,
            "correctness":correctness,
            "fold": explainer.fold_id,
            "counterfactual_label": oracle.predict(counterfactual),
            "counterfactual_adj": counterfactual.data,
            "counterfactual_nodes": counterfactual.node_features,
            "counterfactual_edges": counterfactual.edge_features
        }
        
        with open(results_uri,'w') as dump_file:
            dump_file.write(jsonpickle.encode(info))
        
        return -1
    
    def __create_dirs(self, oracle: Oracle, explainer: Explainer, dataset: Dataset) -> str:
        output_path = join(self._store_path, oracle.context._scope)
        makedirs(output_path, exist_ok=True)

        output_path = join(output_path, dataset.name)
        makedirs(output_path, exist_ok=True)

        output_path = join(output_path, oracle.name)
        makedirs(output_path, exist_ok=True)

        output_path = join(output_path, explainer.name)
        makedirs(output_path, exist_ok=True)
        
        output_path = join(output_path, str(explainer.fold_id))
        makedirs(output_path, exist_ok=True)
        
        output_path = join(output_path, str(oracle.context.run_number))
        makedirs(output_path, exist_ok=True)
        
        return output_path
