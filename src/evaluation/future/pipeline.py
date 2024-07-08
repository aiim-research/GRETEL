import os
import time
from abc import ABCMeta, abstractmethod
import jsonpickle
from typing import List

from src.core.configurable import Configurable
from src.core.explainer_base import Explainer
from src.core.oracle_base import Oracle
from src.utils.context import Context,clean_cfg
from src.utils.logger import GLogger
from src.evaluation.stages.stage import Stage
from src.dataset.instances.base import DataInstance
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.dataset.dataset_base import Dataset


class Pipeline(Configurable, metaclass=ABCMeta):
    """
    This class defines a single full explanation pipeline. It is defined for a particular dataset, oracle and explainer.
    For each instane of the dataset the pipeline creates an explanation object.
    The pipeline is defined with a list of stages that are execute in order. 
    Each stage performs some action and writes information in the explanation object that will be passed to the next stage.
    """

    def __init__(self, 
                 context: Context, 
                 local_config, scope: str, 
                 dataset : Dataset, 
                 oracle: Oracle, 
                 explainer: Explainer, 
                 stages: List[Stage], 
                 results_store_path: str, 
                 run_number=0) -> None:
        
        super().__init__(context=context, local_config=local_config)
        self._logger = GLogger.getLogger()
        self._scope = scope
        self._data = dataset
        self._oracle = oracle
        self._oracle.reset_call_count()
        self._explainer = explainer
        self._results_store_path = results_store_path
        self._stages = stages
        self._run_number = run_number
        self._explanations = []
        
       
        # Building the config file to write into disk
        evaluator_config = {'dataset': clean_cfg(dataset.local_config), 
                            'oracle': clean_cfg(oracle.local_config), 
                            'explainer': clean_cfg(explainer.local_config), 
                            'stages': []}
        
        evaluator_config['scope']=self._scope
        evaluator_config['run_id']=self._run_number
        evaluator_config['fold_id']=self._explainer.fold_id
        evaluator_config['experiment']=dataset.context.conf["experiment"]
        evaluator_config['store_paths']=dataset.context.conf["store_paths"]
        evaluator_config['orgin_config_paths'] = dataset.context.config_file
        
        
        for stage in stages:
            evaluator_config['stages'].append(stage.local_config)
        # creatig the results dictionary with the basic info
        self._results = {}
        self._complete = {'config':evaluator_config, "results":self._results}


    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger


    def init(self):
        super().init()


    @property
    def dataset(self):
        return self._data


    @property
    def explanations(self):
        return self._explanations
    

    @property
    def oracle(self):
        return self._oracle
    

    @property
    def explainer(self):
        return self._explainer


    def run(self):
        for m in self._stages:
            self._results[Context.get_fullname(m)] = []

        # If the explainer was trained then evaluate only on the test set, else evaluate on the entire dataset
        fold_id = self._explainer.fold_id
        if fold_id > -1 :
            test_indices = self.dataset.splits[fold_id]['test']          
            test_set = [i for i in self.dataset.instances if i.id in test_indices]
        else:
            test_set = self.dataset.instances 

        # Evaluating the individual instances of the dataset
        for inst in test_set:
            self._logger.info("Evaluating instance with id %s", str(inst.id))
            explanation = self._real_run(inst)
            self._logger.info('evaluated instance with id %s', str(inst.id))

            self._explanations.append(explanation)

        # Writting the results
        self._logger.info(self._results)
        self.write_results(fold_id)


    def _real_run(self, data_instance: DataInstance):

        explanation = LocalGraphCounterfactualExplanation(context=self.context,
                                                          dataset=self._data, 
                                                          oracle=self._oracle, 
                                                          explainer=self._explainer, 
                                                          input_instance=data_instance, 
                                                          counterfactual_instances=[])
        
        for stage in self._stages:     
                explanation = stage.process(explanation)
                self._results[Context.get_fullname(stage)].append({"id":str(explanation.input_instance.id),
                                                            "value":explanation._stages_info[Context.get_fullname(stage)]})


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