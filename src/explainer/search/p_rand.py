import random
import itertools
import numpy as np
import copy

from src.dataset.instances.graph import GraphInstance
from src.dataset.dataset_base import Dataset
from src.core.explainer_base import Explainer
from src.core.oracle_base import Oracle
from src.core.trainable_base import Trainable

from src.core.factory_base import get_instance_kvargs
from src.utils.cfg_utils import get_dflts_to_of, init_dflts_to_of, inject_dataset, inject_oracle, retake_oracle, retake_dataset


class PRandExplainer(Explainer):

    def check_configuration(self):
        super().check_configuration()

        if not 'p' in self.local_config['parameters']:
            self.local_config['parameters']['p'] = 0.1


    def init(self):
        super().init()

        self.perturbation_percentage = self.local_config['parameters']['p']
        
        
    def explain(self, instance):
        nodes = instance.data.shape[0]

        # all edges (direct graph)
        all_edges = list(itertools.product(list(range(nodes)), repeat=2))
        # filter for only undirected edges
        new_edges = list()
        for edge in all_edges:
            if ((edge[1], edge[0]) not in new_edges) and edge[0] != edge[1]:
                new_edges.append(list(edge))
        new_edges = np.array(new_edges)
        # sample according to perturbation_percentage

        # Creating the instance to return
        adj = copy.deepcopy(instance.data)
        
        sample_index = np.random.choice(list(range(len(new_edges))),
                                         size=int(len(new_edges) * self.perturbation_percentage))
        
        sampled_edges = new_edges[sample_index]
        # switch on/off the sampled edges
        adj[sampled_edges[:,0], sampled_edges[:,1]] = 1 - adj[sampled_edges[:,0], sampled_edges[:,1]]
        adj[sampled_edges[:,1], sampled_edges[:,0]] = 1 - adj[sampled_edges[:,1], sampled_edges[:,0]]
    
        # Encapsulating the perturbating adjacency matrix into a new instance
        result = GraphInstance(id=instance.id,
                               label=0,
                               data=adj,
                               node_features=instance.node_features)
        return result