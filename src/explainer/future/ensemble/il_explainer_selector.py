from typing import List
import numpy as np
import copy

from src.core.explainer_base import Explainer
from src.dataset.dataset_base import Dataset
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



class InstanceLearningExplainerSelector(ExplainerSelector):
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

        # Initializing model related configs
        if 'epochs' not in self.local_config['parameters']:
            self.local_config['parameters']['epochs'] = 200

        if 'batch_size' not in self.local_config['parameters']:
            self.local_config['parameters']['batch_size'] = 32

        if 'optimizer' not in self.local_config['parameters']:
            self.local_config['parameters']['optimizer'] = {'class': 'torch.optim.RMSprop', 
                                                            'parameters': {'lr':0.01}
                                                            }
            
        if 'loss_fn' not in self.local_config['parameters']:
            self.local_config['parameters']['loss_fn'] = {'class': 'torch.nn.CrossEntropyLoss',
                                                          'parameters' : {'reduction': 'mean'}
                                                          }
            
        if 'model' not in self.local_config['parameters']:
            self.local_config['parameters']['model'] = {'class': 'src.oracle.nn.gcn.DownstreamGCN',
                                                        'parameters' : {'num_conv_layers':3,
                                                                        'num_dense_layers':1,
                                                                        'conv_booster':2,
                                                                        'linear_decay':1.8
                                                                        }
                                                        }


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
        relabeled_dataset: Dataset = self.generate_training_dataset(self.dataset)
        loader = relabeled_dataset.get_torch_loader(fold_id=self.fold_id, batch_size=self.batch_size, usage='test')
        
        losses = []
        labels_list, preds = [], []
        for batch in loader:
            batch.batch = batch.batch.to(self.device)
            node_features = batch.x.to(self.device)
            edge_index = batch.edge_index.to(self.device)
            edge_weights = batch.edge_attr.to(self.device)
            labels = batch.y.to(self.device).long()
            
            self.optimizer.zero_grad()  
            pred = self.model(node_features, edge_index, edge_weights, batch.batch)            
            loss = self.loss_fn(pred, labels)
            losses.append(loss.to('cpu').detach().numpy())
            
            labels_list += list(labels.squeeze().long().detach().to('cpu').numpy())
            preds += list(pred.squeeze().detach().to('cpu').numpy())
            
        accuracy = self.accuracy(labels_list, preds)
        self.context.logger.info(f'Test accuracy = {np.mean(accuracy):.4f}')




    def explain(self, instance):

        raise NotImplementedError()

        if not self.best_explainer:
            raise Exception("The explainer was not trained so a base_explainer was not selected")
        
        result = self.best_explainer.explain(instance)
        return result


    def write(self):
        pass
      
    def read(self):
        pass

    def generate_training_dataset(self, original_dataset: Dataset) -> Dataset:
        # Create a copy of the original dataset
        result_dataset = copy.deepcopy(original_dataset)

        # Lets re-label the instance of the new dataset
        for instance in result_dataset.instances:
            explanations = []
            for idx, explainer in enumerate(self.base_explainers):
                exp = explainer.explain(instance)
                exp.producer = explainer
                exp.info['explainer_index'] = idx
                explanations.append(exp)

            # We try to consider only correct explanations
            filtered_explanations = self.explanation_filter.filter(explanations)

            # each instance needs a label even if no explainer was able to produce a correct explanation
            if len(filtered_explanations) < 1:
                filtered_explanations = explanations

            # expand the counterfactuals inside the explanations and iterate over them
            cf_instances = []
            cf_explainers = []
            cf_explaier_indices = []
            for exp in filtered_explanations:
                for cf in exp.counterfactual_instances:
                    # Keep a list with the cf instances
                    cf_instances.append(cf)
                    # Keep a list with the explainer used to generate each counterfactual
                    cf_explainers.append(exp.explainer)
                    # keep a list with the index of the explainer used to generate each counterfactual
                    cf_explaier_indices.append(exp.info['explainer_index']) 
            
            # Build the criteria and gain_direction matrices
            criteria_matrix = np.array(
                [
                    [criteria.calculate(instance, cf, self.oracle, explainer, self.dataset) for criteria in self.criterias]
                    for cf, explainer in zip(cf_instances, cf_explainers)
                ]
            )
            gain_directions = np.array(
                [criteria.gain_direction().value for criteria in self.criterias]
            )

            # Find the best counterfactual according to the multiple criterias
            best_index = find_best(criteria_matrix,
                                   gain_directions,
                                   self.distance.calculate
                                   )
            
            # Getting the index of the explainer that produced the best results
            best_explainer = cf_explaier_indices[best_index]

            # change the labe for the id of the best explainer
            instance.label = best_explainer

        # Returning the new dataset
        return result_dataset
