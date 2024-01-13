import os
import time
from abc import ABC

import jsonpickle
from scipy import rand

from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.explainer_base import Explainer
from src.core.oracle_base import Oracle
from src.utils.cfgnnexplainer.utils import safe_open
from src.utils.context import Context,clean_cfg
from src.utils.logger import GLogger


class Evaluator(ABC):
    _logger = GLogger.getLogger()

    def __init__(self,scope, data, oracle: Oracle, explainer: Explainer, evaluation_metrics, results_store_path, run_number=0) -> None:
        super().__init__()
        self._scope = scope
        self._name = 'Evaluator_for_' + explainer.name + '_using_' + oracle.name
        self._data = data
        self._oracle = oracle
        self._oracle.reset_call_count()
        self._explainer = explainer
        self._results_store_path = results_store_path
        self._evaluation_metrics = evaluation_metrics
        self._run_number = run_number
        self._explanations = []
        
       
        

        # Building the config file to write into disk
        evaluator_config = {'dataset': clean_cfg(data.local_config), 'oracle': clean_cfg(oracle.local_config), 'explainer': clean_cfg(explainer.local_config), 'metrics': []}
        evaluator_config['scope']=self._scope
        evaluator_config['run_id']=self._run_number
        evaluator_config['fold_id']=self._explainer.fold_id
        evaluator_config['experiment']=data.context.conf["experiment"]
        evaluator_config['store_paths']=data.context.conf["store_paths"]
        evaluator_config['orgin_config_paths'] = data.context.config_file
        
        
        for metric in evaluation_metrics:
            evaluator_config['metrics'].append(metric._config_dict)
        # creatig the results dictionary with the basic info
        self._results = {}
        self._complete = {'config':evaluator_config, "results":self._results}

        

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def dataset(self):
        return self._data

    @dataset.setter
    def dataset(self, new_dataset):
        self._data = new_dataset

    @property
    def explanations(self):
        return self._explanations

    @explanations.setter
    def explanations(self, new_explanations_list):
        self._explanations = new_explanations_list


    def get_instance_explanation_pairs(self):
        # Check if the explanations were generated already
        if len(self.explanations) < 1:
            return None

        # iterates over the original instances and the explanations
        n_ins = len(self.dataset.instances)
        result = []
        for i in range(0, n_ins):
            result.append((self.dataset.instances[i], self.explanations[i]))

        return result


    def get_instance_and_counterfactual_classifications(self):
        # Check if the explanations were generated already
        if len(self.explanations) < 1:
            return None

        # iterates over the original instances and the explanations
        n_ins = len(self.dataset.instances)
        result = []
        for i in range(0, n_ins):
            label_inst = self._oracle.predict(self.dataset.instances[i])
            label_cf = self._oracle.predict(self.explanations[i])
            self._oracle._call_counter -= 2 

            result.append({'instance_id': self.dataset.instances[i].id,
                             'ground_truth_label': self.dataset.instances[i].graph_label,
                             'instance_label': label_inst,
                             'counterfactual_label': label_cf})

        return result


    def evaluate(self):
        for m in self._evaluation_metrics:
            self._results[Context.get_fullname(m)] = []

        # If the explainer was trained then evaluate only on the test set, else evaluate on the entire dataset
        fold_id = self._explainer.fold_id
        if fold_id > -1 :
            test_indices = self.dataset.splits[fold_id]['test']          
            test_set = [i for i in self.dataset.instances if i.id in test_indices]
        else:
            test_set = self.dataset.instances 

        for inst in test_set:
            self._logger.info("Evaluating instance with id %s", str(inst.id))

            for metric in self._evaluation_metrics:
                if(metric._special):
                    val, counterfactual = metric.evaluate(inst, None, self._oracle,self._explainer,self._data)
                    self._results[Context.get_fullname(metric)].append({"id":str(inst.id),"value":val})
                    self._explanations.append(counterfactual)

            self._real_evaluate(inst, counterfactual,self._oracle,self._explainer,self._data)
            self._logger.info('evaluated instance with id %s', str(inst.id))

        self._logger.info(self._results)
    
        self.write_results(fold_id)


    def _real_evaluate(self, instance, counterfactual, oracle = None, explainer=None, dataset=None):
        is_alt = False
        if (oracle is None):
            is_alt = True
            oracle = self._oracle

        for metric in self._evaluation_metrics:
            if(not metric._special):        
                m_result = metric.evaluate(instance, counterfactual, oracle, explainer,dataset)
                self._results[Context.get_fullname(metric)].append({"id":str(instance.id),"value":m_result})


    def write_results(self,fold_id):
        hash_info = {"scope":self._scope,
                      "dataset":self._data.name,
                      "oracle":self._oracle.name,
                      "explainer":self._explainer.name
                      }
        
        self._complete['hash_ids']=hash_info

        output_path = os.path.join(self._results_store_path, self._scope)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        
        output_path = os.path.join(output_path, self._data.name)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        output_path = os.path.join(output_path, self._oracle.name)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        output_path = os.path.join(output_path, self._explainer.name)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        results_uri = os.path.join(output_path, 'results_' + str(fold_id) + '_'+ str(self._run_number)+'.json')

        with open(results_uri, 'w') as results_writer:
            results_writer.write(jsonpickle.encode(self._complete))

