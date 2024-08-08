from typing import List
import numpy as np
import copy
import torch
from torch_geometric.loader import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import accuracy_score
import random
from torch.utils.data import Subset

from src.core.explainer_base import Explainer
from src.dataset.dataset_base import Dataset
from src.explainer.future.ensemble.explainer_selector_base import ExplainerSelector
from src.core.factory_base import get_class, get_instance_kvargs
from src.utils.cfg_utils import  inject_dataset, inject_oracle
from src.utils.cfg_utils import init_dflts_to_of
from src.dataset.utils.dataset_torch import TorchGeometricDataset

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
    This explainer learns which base explainer produces the best explanation for an instance given its features and structure
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
        self.local_config['parameters']['epochs'] = self.local_config['parameters'].get('epochs', 200)
        self.local_config['parameters']['batch_size'] = self.local_config['parameters'].get('batch_size', 4)
        self.local_config['parameters']['early_stopping_threshold'] = self.local_config['parameters'].get('early_stopping_threshold', None)

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
        self.local_config['parameters']['model']['parameters']['node_features'] = self.dataset.num_node_features()
        self.local_config['parameters']['model']['parameters']['n_classes'] = len(self.local_config['parameters']['explainers'])


    def init(self):
        super().init()
        
        # Initializing base explainers.........................................
        self.base_explainers = [ get_instance_kvargs(exp['class'],
                    {'context':self.context,'local_config':exp}) for exp in self.local_config['parameters']['explainers']]
        
        # Initializing evaluation related objects...............................
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

        # Initializing model-related objects..................................
        self.epochs = self.local_config['parameters']['epochs']
        self.batch_size = self.local_config['parameters']['batch_size']
        self.early_stopping_threshold = self.local_config['parameters']['early_stopping_threshold']

        self.model = get_instance_kvargs(self.local_config['parameters']['model']['class'],
                                   self.local_config['parameters']['model']['parameters'])
        
        self.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})
        
        self.loss_fn = get_instance_kvargs(self.local_config['parameters']['loss_fn']['class'],
                                           self.local_config['parameters']['loss_fn']['parameters'])
        
        self.lr_scheduler =  lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5, total_iters=self.epochs)

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.model.to(self.device) 
        
        self.patience = 0      


    def real_fit(self):
        # Create the dataset where the label is the index of the base explainer that produced the best results
        relabeled_dataset: Dataset = self.generate_explainer_prediction_dataset(self.dataset)

        # creating a torch dataloader with the train instances of the given fold
        instances = self.dataset.get_torch_instances(fold_id=self.fold_id, usage='test')
        train_loader, val_loader = None, None

        if self.early_stopping_threshold:
            num_instances = len(self.dataset.instances)
            # get 5% of training instances and reserve them for validation
            indices = list(range(num_instances))
            random.shuffle(indices)
            val_size = max(int(.05 * len(indices)), self.batch_size)
            train_size = len(indices) - val_size
            # get the training instances
            train_instances = Subset(instances, indices[:train_size - 1])
            val_instances = Subset(instances, indices[train_size:])
            # get the train and validation loaders
            train_loader = DataLoader(train_instances, batch_size=self.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_instances, batch_size=self.batch_size, shuffle=True, drop_last=True)
        else:
            train_loader = DataLoader(instances, batch_size=self.batch_size, shuffle=True, drop_last=True)    

        best_loss = [0,0]

        # Train for the set number of epochs
        for epoch in range(self.epochs):
            losses, preds, labels_list = [], [], []
            # Set the model in train mode
            self.model.train()
            # Iterate over the batches in the data
            for batch in train_loader:
                # Get the batch data
                batch.batch = batch.batch.to(self.device)
                node_features = batch.x.to(self.device)
                edge_index = batch.edge_index.to(self.device)
                edge_weights = batch.edge_attr.to(self.device)
                labels = batch.y.to(self.device).long()
                
                self.optimizer.zero_grad()
                
                # Get the prediction for the batch and calculate the loss
                pred = self.model(node_features, edge_index, edge_weights, batch.batch)
                loss = self.loss_fn(pred, labels)
                losses.append(loss.to('cpu').detach().numpy())
                # Backpropagate
                loss.backward()
                
                # Add the batch results to the epoch results
                labels_list += list(labels.squeeze().long().detach().to('cpu').numpy())
                preds += list(pred.squeeze().detach().to('cpu').numpy())
               
                self.optimizer.step()

            # Calculate the accuracy at the given epoch using all batch results
            accuracy = accuracy_score(labels_list, np.argmax(preds, axis=1))
            self.context.logger.info(f'epoch = {epoch} ---> loss = {np.mean(losses):.4f}\t accuracy = {accuracy:.4f}')
            self.lr_scheduler.step()

            # check if we need to do early stopping
            if self.early_stopping_threshold and len(val_loader) > 0:
                self.model.eval()
                val_losses, val_labels, val_preds = [], [], []
                with torch.no_grad():
                    for batch in val_loader:
                        batch.batch = batch.batch.to(self.device)
                        node_features = batch.x.to(self.device)
                        edge_index = batch.edge_index.to(self.device)
                        edge_weights = batch.edge_attr.to(self.device)
                        labels = batch.y.to(self.device).long()

                        pred = self.model(node_features, edge_index, edge_weights, batch.batch)
                        loss = self.loss_fn(pred, labels)
                        
                        val_labels += list(labels.squeeze().to('cpu').numpy())
                        val_preds += list(pred.squeeze().to('cpu').numpy())
                        
                        val_losses.append(loss.item())
                        
                    best_loss.pop(0)
                    var_loss = np.mean(val_losses)
                    best_loss.append(var_loss)
                    
                    accuracy = accuracy_score(val_labels, np.argmax(val_preds, axis=1))
                    self.context.logger.info(f'epoch = {epoch} ---> var_loss = {var_loss:.4f}\t var_accuracy = {accuracy:.4f}')
                
                if abs(best_loss[0] - best_loss[1]) < self.early_stopping_threshold:
                    self.patience += 1
                    
                    if self.patience == 4:
                        self.context.logger.info(f"Early stopped training at epoch {epoch}")
                        break  # terminate the training loop



    def explain(self, instance):
        # use the model to predict the best based explainer for the instance
        predicted_exp_idx = self.predict_explainer(instance)
        best_explainer = self.base_explainers[predicted_exp_idx]

        # use the selected explainer to produce an explanation
        result_explanation = best_explainer.explain(instance)
        return result_explanation
    

    def generate_explainer_prediction_dataset(self, original_dataset: Dataset) -> Dataset:
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
    

    def predict_explainer(self, data_inst):
        # Set the model in evaluation mode
        self.model.eval()

        data_inst = TorchGeometricDataset.to_geometric(data_inst)
        node_features = data_inst.x.to(self.device)
        edge_index = data_inst.edge_index.to(self.device)
        edge_weights = data_inst.edge_attr.to(self.device)
        
        exp_probs = self.model(node_features,edge_index,edge_weights, None).cpu().squeeze()
        return torch.argmax(exp_probs).item()
    
