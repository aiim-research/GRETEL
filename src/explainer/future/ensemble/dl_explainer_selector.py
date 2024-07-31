from typing import List
import numpy as np

from src.core.explainer_base import Explainer
from src.explainer.future.ensemble.explainer_selector_base import ExplainerSelector
from src.core.factory_base import get_class, get_instance_kvargs
from src.utils.cfg_utils import  inject_dataset, inject_oracle
from src.utils.cfg_utils import init_dflts_to_of

from src.explainer.future.ensemble.aggregators.multi_criteria.algorithm import find_best
from src.explainer.future.ensemble.aggregators.multi_criteria.criterias.base_criteria import (
    BaseCriteria,
)
from src.explainer.future.ensemble.aggregators.multi_criteria.distances.base_distance import (
    BaseDistance,
)
from src.future.explanation.local.graph_counterfactual import (
    LocalGraphCounterfactualExplanation,
)



class DatasetLevelExplainerSelector(ExplainerSelector):
    """
    This explainer determines wich base explainer is the best for a given dataset and uses it for explaining all instances
    """

    def check_configuration(self):
        super().check_configuration()

        for exp in self.local_config['parameters']['explainers']:
            exp['parameters']['fold_id'] = self.local_config['parameters']['fold_id']
            # In any case we need to inject oracle and the dataset to the model
            inject_dataset(exp, self.dataset)
            inject_oracle(exp, self.oracle)

        # Initializing the distance for the multi-criteria selection
        default_distance = "src.explainer.future.ensemble.aggregators.multi_criteria.distances.euclidean_distance.EuclideanDistance"
        init_dflts_to_of(self.local_config, "distance", default_distance)


    def init(self):
        super().init()
        self.best_explainer = None
        
        # Initializing base explainers
        self.base_explainers = [ get_instance_kvargs(exp['class'],
                    {'context':self.context,'local_config':exp}) for exp in self.local_config['parameters']['explainers']]
        
        # Inizializing performance criterias
        self.criterias: List[BaseCriteria] = [
            get_instance_kvargs(
                exp["class"], {"context": self.context, "local_config": exp}
            )
            for exp in self.local_config["parameters"]["criterias"]
        ]

        # Inizializing distance
        self.distance: BaseDistance = get_instance_kvargs(
            self.local_config["parameters"]["distance"]["class"],
            {
                "context": self.context,
                "local_config": self.local_config["parameters"]["distance"],
            },
        )

    def real_fit(self):
        training_data_indices = self.dataset.get_split_indices(self.fold_id)['train']
        training_data = [self.dataset.instances[idx] for idx in training_data_indices]

        explainer_scores = np.zeros(len(self.base_explainers)).tolist()
        for instance in training_data:
            explanations = []
            for idx, explainer in enumerate(self.base_explainers):
                exp = explainer.explain(instance)
                exp.producer = explainer
                exp.info['explainer_index'] = idx
                explanations.append(exp)


            
            cf_instances = []
            cf_explainers = []
            cf_explaier_indices = []

            for exp in explanations:
                for cf in exp.counterfactual_instances:
                    cf_instances.append(cf)
                    cf_explainers.append(exp.explainer)
                    cf_explaier_indices.append(exp.info['explainer_index'])
            
            criteria_matrix = np.array(
                [
                    [criteria.calculate(instance, cf, self.oracle, explainer, self.dataset) for criteria in self.criterias]
                    for cf, explainer in zip(cf_instances, cf_explainers)
                ]
            )
            gain_directions = np.array(
                [criteria.gain_direction().value for criteria in self.criterias]
            )

            best_index = find_best(
            criteria_matrix,
            gain_directions,
            self.distance.calculate,
            )
            # Getting the explainer that produced the best results
            best_cf = cf_instances[best_index]
            best_explainer = cf_explaier_indices[best_index]
            # Updating the explainer score in the record
            explainer_scores[best_explainer] += 1

        best_exp_idx = explainer_scores.index(max(explainer_scores))
        self.best_explainer = self.base_explainers[best_exp_idx]


    def explain(self, instance):

        if not self.best_explainer:
            raise Exception("The explainer was not trained so a base_explainer was not selected")
        
        result = self.best_explainer.explain(instance)
        return result


    def write(self):
        pass
      
    def read(self):
        pass