from types import SimpleNamespace
from typing import List

import torch

from src.core.oracle_base import Oracle
from src.utils.samplers.abstract_sampler import Sampler
from src.dataset.instances.graph import GraphInstance


class Bernoulli(Sampler):
    
    def __init__(self, sampling_iterations=1):
        super().__init__()
        self._name = 'bernoulli'
        self.sampling_iterations = sampling_iterations
          
    def sample(self, instance: GraphInstance, oracle: Oracle, **kwargs) -> List[GraphInstance]:
        kwargs = SimpleNamespace(**kwargs)
        edge_probs = kwargs.edge_probabilities
        embedded_features = kwargs.embedded_features

        pred_label = oracle.predict(instance)
        edge_probs = edge_probs.get(pred_label)
        node_features = embedded_features.get(pred_label).cpu().numpy()
        
        for _ in range(self.sampling_iterations):
            cf_instance = self.__sample(instance, node_features, edge_probs)
            if oracle.predict(cf_instance) != pred_label:
                return cf_instance
            
        return None 
                
    def __sample(self, instance, features, probabilities):
        adj = torch.bernoulli(probabilities)
        selected_edges = torch.nonzero(adj)
        
        cf_candidate = GraphInstance(id=instance.id,
                             label=1-instance.label,
                             data=adj.numpy(),
                             node_features=features,
                             edge_weights=probabilities[selected_edges[:, 0], selected_edges[:, 1]].numpy())
        
        for manipulator in instance._dataset.manipulators:
            manipulator._process_instance(cf_candidate)
        
        return cf_candidate