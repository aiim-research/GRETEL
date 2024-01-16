import itertools
from types import SimpleNamespace
from typing import List

import numpy as np
import torch

from src.core.oracle_base import Oracle
from src.dataset.instances.graph import GraphInstance

from src.utils.samplers.abstract_sampler import Sampler


class PositiveAndNegativeEdgeSampler(Sampler):
    
    def __init__(self, sampling_iterations):
        super().__init__()
        self._name = 'positive_and_negative_edge_sampler'
        self.sampling_iterations = sampling_iterations
          
    def sample(self, instance: GraphInstance, oracle: Oracle, **kwargs) -> List[GraphInstance]:
        kwargs = SimpleNamespace(**kwargs)
        edge_probs = kwargs.edge_probabilities
        embedded_features = kwargs.embedded_features
        edge_list = self.__get_edges(instance)
        edge_num = instance.num_edges
        # if I have any edges in the original instance
        if edge_list.size:
            pred_label = oracle.predict(instance)
            edge_probs = edge_probs.get(pred_label)
            node_features = embedded_features.get(pred_label).cpu().numpy()
            cf_instance = self.__sample(instance, node_features, edge_probs[edge_list[0,:], edge_list[1,:]], edge_list, num_samples=edge_num)
            if oracle.predict(cf_instance) != pred_label:
                return cf_instance
            else:
                for _ in range(self.sampling_iterations):
                    missing_edges = self.__negative_edges(self.__get_edges(cf_instance), instance.num_nodes)
                    # if there are any missing edges to sample from
                    if missing_edges.numel():
                        probs = edge_probs[missing_edges[0,:], missing_edges[1,:]]
                        #probs = torch.from_numpy(np.array([1 / len(missing_edges) for _ in range(len(missing_edges))]))
                        cf_instance = self.__sample(cf_instance, node_features, probs, missing_edges)
                        if oracle.predict(cf_instance) != pred_label:
                            return cf_instance
        return None 
                
    def __negative_edges(self, edges, num_vertices):
        all_edges = set(itertools.product(range(num_vertices), repeat=2))
        edges = set(zip(edges[0], edges[1]))
        diff = list(all_edges.difference(edges))
        t = torch.from_numpy(self.__build_array_from_tuples(diff))
        return t
  
    def __sample(self, instance, features, probabilities, edge_list, num_samples=1):
        n_nodes = instance.num_nodes
        adj = torch.zeros((n_nodes, n_nodes)).double()
        selected_indices = torch.multinomial(probabilities, num_samples=num_samples, replacement=False).numpy()
        for edge in edge_list[:,selected_indices].T:
            adj[edge[0], edge[1]] = 1
        return GraphInstance(id=instance.id, label=1-instance.label, data=adj.numpy(), node_features=features)      
    
    def __get_edges(self, instance):
        indices = np.nonzero(instance.data)
        return np.array([indices[0], indices[1]])
    
    def __build_array_from_tuples(self, list_of_tuples):
        first_elements = np.array([t[0] for t in list_of_tuples])
        second_elements = np.array([t[1] for t in list_of_tuples])
        return np.array([first_elements, second_elements])
