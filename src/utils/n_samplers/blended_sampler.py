from types import SimpleNamespace
from typing import List

import numpy as np
import torch


#TODO: Bardh Please Revise and FIX IT
from src.dataset.instances import DataInstance
from src.dataset.data_instance_features import DataInstanceWFeaturesAndWeights
from src.core.oracle_base import Oracle
from src.utils.n_samplers.abstract_sampler import Sampler


class BlendedSampler(Sampler):
    
    def __init__(self, sampling_iterations):
        super().__init__()
        self._name = 'blended_sampler'
        self.sampling_iterations = sampling_iterations      
          
    def sample(self, instance: DataInstance, oracle: Oracle, **kwargs) -> List[DataInstance]:
        kwargs = SimpleNamespace(**kwargs)
        
        edge_probs = kwargs.edge_probabilities
        embedded_features = kwargs.embedded_features
        edge_list = kwargs.edge_index
        
        sum_tensor = list(edge_probs.values())[0]
        # average the features
        average_features = torch.zeros_like(next(iter(embedded_features.values())))
        for tensor in embedded_features.values():
            average_features += tensor
        average_features /= len(embedded_features.values())

        edge_num = instance.graph.number_of_edges()
        rows, cols = edge_list[0], edge_list[1]
        result = sum_tensor[rows, cols]
        
        cf_instance = self.__sample(instance, average_features, result, edge_list, num_samples=edge_num) 
        if oracle.predict(cf_instance) != oracle.predict(instance):
            return cf_instance
        else:
            # get the "negative" edges
            missing_edges = self.__negative_edges(edge_list, instance.graph.number_of_nodes())
            edge_probs = sum_tensor[missing_edges[0], missing_edges[1]]
            # check sampling for sampling_iterations
            # and see if we find a valid counterfactual
            for _ in range(self.sampling_iterations):
                cf_instance = self.__sample(cf_instance, average_features, edge_probs, missing_edges, graph=cf_instance)
                if oracle.predict(cf_instance) != oracle.predict(instance):
                    return cf_instance
        return None 
  
    def __negative_edges(self, edges, num_vertices):
        i, j = np.triu_indices(num_vertices, k=1)
        all_edges = set(list(zip(i, j)))
        edges = set([tuple(x) for x in edges])
        return torch.from_numpy(np.array(list(all_edges.difference(edges)))).T
    
    def __sample(self, instance: DataInstance, features, probabilities, edge_list, num_samples=1, graph=None):       
        n_nodes = instance.graph.number_of_nodes()
        adj = torch.zeros((n_nodes, n_nodes)).double()
        weights = torch.zeros((n_nodes, n_nodes)).double()
        ##################################################
        cf_instance = DataInstanceWFeaturesAndWeights(id=instance.id)
        selected_indices = torch.multinomial(probabilities, num_samples=num_samples, replacement=True).numpy()
        adj[edge_list[0,selected_indices],edge_list[1,selected_indices]] = 1
        adj = adj + adj.T - torch.diag(torch.diag(adj))
        
        weights[edge_list[0,selected_indices], edge_list[1,selected_indices]] = probabilities[selected_indices]
        weights = weights + weights.T - torch.diag(torch.diag(weights))
        
        if graph:
            adj = adj + graph.to_numpy_array()
            weights = weights + graph.weights
        
        cf_instance.from_numpy_array(adj.numpy())
        cf_instance.weights = weights.numpy()
        cf_instance.features = features.numpy()
        
        return cf_instance