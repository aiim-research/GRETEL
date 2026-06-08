import json
import os
import time
from abc import ABC

import jsonpickle
import numpy as np
import pickle

from src.core.configurable import Configurable
from src.core.explainer_base import Explainer
from src.core.oracle_base import Oracle
from src.dataset.dataset_base import Dataset
from src.utils.cfgnnexplainer.utils import safe_open
from src.utils.context import Context,clean_cfg
from src.utils.logger import GLogger
from src.utils.cfg_utils import retake_dataset, retake_oracle, retake_explainer, retake_scope, retake_results_store_path, retake_run_number
from src.core.factory_base import get_class, get_instance_kvargs
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation


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

        self._pipeline = get_instance_kvargs(self.local_config['parameters']['pipeline']['class'], 
                                             {'context':self.context,'local_config':self.local_config['parameters']['pipeline']})
        
        # Building the config file to write into disk
        evaluator_config = {'dataset': clean_cfg(self._dataset.local_config), 
                            'oracle': clean_cfg(self._oracle.local_config), 
                            'explainer': clean_cfg(self._explainer.local_config), 
                            'stages': []}
        
        evaluator_config['scope'] = self._scope
        evaluator_config['run_id'] = self._run_number
        evaluator_config['fold_id'] = self._explainer.fold_id
        evaluator_config['experiment'] = self.context.conf["experiment"]
        evaluator_config['store_paths'] = self.context.conf["store_paths"]
        evaluator_config['orgin_config_paths'] = self._dataset.context.config_file
        
        
        for stage in self._pipeline.stages:
            evaluator_config['stages'].append(stage.local_config)

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
        # Initializing the list with the results of each stage (For keeping backward-compatibility)
        for stage in self._pipeline.stages:
            self._results[Context.get_fullname(stage)] = []

        # If the explainer was trained then evaluate only on the test set, else evaluate on the entire dataset
        fold_id = self._explainer.fold_id
        if fold_id > -1 :
            test_indices = self.dataset.splits[fold_id]['test']         
            test_set = [i for i in self.dataset.instances if i.id in test_indices]
        else:
            test_set = self.dataset.instances 

        for inst in test_set:
            self._logger.info("Evaluating instance with id %s", str(inst.id))

            self._real_evaluate(inst)
            self._logger.info('evaluated instance with id %s', str(inst.id))



        self._logger.info(self._results)
        self.write_results(fold_id)


    def _real_evaluate(self, instance):

        # Creating an empty explanation for the pipeline
        explanation = LocalGraphCounterfactualExplanation(context=self.context,
                                                     dataset=self.dataset,
                                                     oracle=self.oracle,
                                                     explainer=self.explainer,
                                                     input_instance=instance,
                                                     counterfactual_instances=[]
                                                     )
        # Pass the instance by the pipeline
        explanation = self._pipeline.process(explanation)

        # Store the results in a backwards-compatible way
        for stage in self._pipeline.stages:
                self._results[Context.get_fullname(stage)].append({"id":str(explanation.input_instance.id),
                                                                   "value": explanation.stages_info[Context.get_fullname(stage)]})
        # Store the explanation internally
        self._explanations.append(explanation)

        # Persist per-instance input + counterfactual + metrics so downstream
        # analyses (Exps. 5/6/7 of REVISION_EXPERIMENTS.md) can run offline.
        try:
            self._dump_explanation_json(explanation, self._explainer.fold_id)
        except Exception as e:
            self._logger.warning("Per-instance JSON dump failed for %s: %s",
                                 str(explanation.input_instance.id), e)

    def _instance_to_dict(self, instance):
        """Compact, JSON-ready view of a GraphInstance.

        Edges are stored as an undirected upper-triangle list (or full list
        when ``directed=True``) so dumps stay greppable for small graphs and
        manageable for large ones. ``node_features`` are dropped when they
        carry no real information (the GraphInstance default is a zero column
        per node)."""
        data = instance.data
        nz = np.nonzero(data)
        if instance.directed:
            edges = [[int(i), int(j)] for i, j in zip(nz[0], nz[1])]
        else:
            edges = [[int(i), int(j)] for i, j in zip(nz[0], nz[1]) if int(i) < int(j)]

        nf = getattr(instance, "node_features", None)
        if nf is not None and hasattr(nf, "shape") and nf.shape[1] > 1:
            node_features = nf.tolist()
        else:
            node_features = None

        return {
            "id": str(instance.id),
            "label": int(instance.label) if instance.label is not None else None,
            "num_nodes": int(data.shape[0]),
            "directed": bool(getattr(instance, "directed", False)),
            "edges": edges,
            "node_features": node_features,
        }

    def _dump_explanation_json(self, explanation, fold_id):
        """Write ``cf_<instance_id>.json`` with (input, counterfactual, metrics).

        Skips silently when no counterfactual was produced (still writes the
        input + metrics so failed cases are visible to Exp. 5's instance
        counts)."""
        output_dir = os.path.join(self._results_store_path, self._scope,
                                  self._dataset.name, self._oracle.name,
                                  self._explainer.name, "cf_per_instance",
                                  f"fold_{fold_id}")
        os.makedirs(output_dir, exist_ok=True)

        metrics = {}
        for stage_name, value in (explanation.stages_info or {}).items():
            short = stage_name.rsplit(".", 1)[-1]
            metrics[short] = value

        ctfs = explanation.counterfactual_instances or []
        payload = {
            "id": str(explanation.input_instance.id),
            "fold_id": int(fold_id),
            "input": self._instance_to_dict(explanation.input_instance),
            "counterfactual": self._instance_to_dict(ctfs[0]) if ctfs else None,
            "metrics": metrics,
        }

        path = os.path.join(output_dir, f"cf_{explanation.input_instance.id}.json")
        with open(path, "w") as f:
            json.dump(payload, f, default=self._json_default)

    @staticmethod
    def _json_default(o):
        """JSON fallback for numpy scalars/arrays leaked into metrics."""
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


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


    def pickle_explanations(self, store_path):
        if len(self._explanations) < 1:
            raise Exception('Trying to pickle an empty explanations list')

        # Ensure the store_path exists
        os.makedirs(store_path, exist_ok=True)

        # Define the full path for the pickle file
        pickle_file_path = os.path.join(store_path, self._explainer.name + '.pkl')

        # Pickle the list into the specified file
        with open(pickle_file_path, 'wb') as pickle_file:
            for exp in self._explanations:
                exp.input_instance._dataset = None
                for inst in exp.counterfactual_instances:
                    inst._dataset = None
                    
            inst_cf_pairs = [(exp.input_instance, exp.counterfactual_instances[0]) for exp in self._explanations]
            pickle.dump(inst_cf_pairs, pickle_file)