import random
import itertools
import numpy as np
import copy
import sys

from src.dataset.instances.graph import GraphInstance
from src.dataset.dataset_base import Dataset
from src.core.explainer_base import Explainer
from src.core.oracle_base import Oracle
from src.core.trainable_base import Trainable

from src.core.factory_base import get_instance_kvargs
from src.utils.cfg_utils import get_dflts_to_of, init_dflts_to_of, inject_dataset, inject_oracle, retake_oracle, retake_dataset
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation


class OvershootExplainer(Explainer):

    def check_configuration(self):
        super().check_configuration()

        if not 'p' in self.local_config['parameters']:
            self.local_config['parameters']['p'] = 0.1

        dst_metric='src.evaluation.evaluation_metric_ged.GraphEditDistanceMetric'  

        #Check if the distance metric exist or build with its defaults:
        init_dflts_to_of(self.local_config, 'distance_metric', dst_metric)


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
        result_instance = GraphInstance(id=instance.id,
                               label=0,
                               data=adj,
                               node_features=instance.node_features)
        
        # Encapsulating the instance into an explanation
        result = LocalGraphCounterfactualExplanation(
                    context=self.context,
                    dataset=self.dataset,
                    oracle=self.oracle,
                    explainer=self,
                    input_instance=instance,
                    counterfactual_instances=[result_instance]
                )
    
        return result
    

    def overshoot(self, instance):
        """
        This method search for the closest counterfactual in the dataset
        """
        input_label = self.oracle.predict(instance)

        # if the method does not find a counterfactual example returns the original graph
        min_ctf = instance

        # Iterating over all the instances of the dataset
        min_ctf_dist = sys.float_info.max
        for ctf_candidate in self.dataset.instances:
            candidate_label = self.oracle.predict(ctf_candidate)

            if input_label != candidate_label:
                ctf_distance = self.distance_metric.evaluate(instance, ctf_candidate, self.oracle)
                
                if ctf_distance < min_ctf_dist:
                    min_ctf_dist = ctf_distance
                    min_ctf = ctf_candidate

        result = copy.deepcopy(min_ctf)
        result.id = instance.id

        return result