import os
import time
from abc import ABC
import jsonpickle

from src.core.configurable import Configurable
from src.core.explainer_base import Explainer
from src.core.oracle_base import Oracle
from src.dataset.dataset_base import Dataset
from src.utils.cfgnnexplainer.utils import safe_open
from src.utils.context import Context,clean_cfg
from src.utils.logger import GLogger
from src.utils.cfg_utils import retake_dataset, retake_oracle, retake_explainer, retake_scope, retake_results_store_path, retake_run_number
from src.core.factory_base import get_class, get_instance_kvargs


class Evaluator(Configurable):
    
    def __init__(self, context: Context, local_config) -> None:
        # Intitializing basic fields
        self._dataset = retake_dataset(local_config)
        self._oracle = retake_oracle(local_config)
        self._oracle.reset_call_count()
        self._explainer = retake_explainer(local_config)
        self._scope = retake_scope(local_config)
        self._results_store_path = retake_results_store_path(local_config)
        self._run_number = retake_run_number(local_config)
        self._logger = GLogger.getLogger()

        self._pipeline = None
        self._explanations = []
        self._results = None
        self._complete = None

        super().__init__(context=context, local_config=local_config)


    def check_configuration(self):
        
        
        super().check_configuration()
        self.logger= self.context.logger


    def init(self):
        super().init()

        self._pipeline = get_instance_kvargs(self.local_config['parameters']['pipeline']['class'], {'context':self.context,'local_config':self.local_config['parameters']['pipeline']})
        
        # Building the config file to write into disk
        evaluator_config = {'dataset': clean_cfg(self._dataset.local_config), 
                            'oracle': clean_cfg(self._oracle.local_config), 
                            'explainer': clean_cfg(self._explainer.local_config), 
                            'stages': []}
        evaluator_config['scope']=self._scope
        evaluator_config['run_id']=self._run_number
        evaluator_config['fold_id']=self._explainer.fold_id
        evaluator_config['experiment']=dataset.context.conf["experiment"]
        evaluator_config['store_paths']=dataset.context.conf["store_paths"]
        evaluator_config['orgin_config_paths'] = dataset.context.config_file
        
        
        for metric in evaluation_metrics:
            evaluator_config['evaluation_metrics'].append(metric.local_config)
        # creatig the results dictionary with the basic info
        self._results = {}
        self._complete = {'config':evaluator_config, "results":self._results}


    @property
    def dataset(self):
        return self._dataset


    @property
    def explanations(self):
        return self._explanations
    

    @property
    def oracle(self):
        return self._oracle
    

    @property
    def explainer(self):
        return self._explainer


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

            start_time = time.time()
            explanation = self._explainer.explain(inst)
            end_time = time.time()
            explanation.explanation_runtime = end_time - start_time
            self._explanations.append(explanation)

            self._real_evaluate(explanation)
            self._logger.info('evaluated instance with id %s', str(inst.id))

        self._logger.info(self._results)
    
        self.write_results(fold_id)


    def _real_evaluate(self, explanation):
        for metric in self._evaluation_metrics:     
                m_result = metric.evaluate(explanation)
                self._results[Context.get_fullname(metric)].append({"id":str(explanation.input_instance.id),"value":m_result})


    def write_results(self,fold_id):
        hash_info = {"scope":self._scope,
                      "dataset":self._dataset.name,
                      "oracle":self._oracle.name,
                      "explainer":self._explainer.name
                      }
        
        self._complete['hash_ids']=hash_info

        output_path = os.path.join(self._results_store_path, self._scope)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        
        output_path = os.path.join(output_path, self._dataset.name)
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

