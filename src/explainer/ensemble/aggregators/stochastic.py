from typing import List

import torch

from src.dataset.instances.graph import GraphInstance
from src.explainer.ensemble.aggregators.base import ExplanationAggregator
from src.utils.samplers.abstract_sampler import Sampler
import numpy as np

from src.core.factory_base import get_instance_kvargs
from src.utils.cfg_utils import init_dflts_to_of


class StochasticAggregator(ExplanationAggregator):

    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger

        dst_metric = 'src.evaluation.evaluation_metric_ged.GraphEditDistanceMetric'
        dflt_sampler = 'src.utils.samplers.bernoulli.Bernoulli'

        #Check if the distance metric exist or build with its defaults:
        init_dflts_to_of(self.local_config, 'distance_metric', dst_metric)

        init_dflts_to_of(self.local_config,
                         'sampler',
                         dflt_sampler,
                         sampling_iterations=500)


    def init(self):
        super().init()
        
        self.distance_metric = get_instance_kvargs(self.local_config['parameters']['distance_metric']['class'], 
                                                   self.local_config['parameters']['distance_metric']['parameters'])
        
        self.sampler: Sampler = get_instance_kvargs(self.local_config['parameters']['sampler']['class'],
                                                    self.local_config['parameters']['sampler']['parameters'])
        

    def aggregate(self, instance: GraphInstance, explanations: List[GraphInstance]):
        label = self.oracle.predict(instance)
        
        edge_freq_matrix = np.zeros_like(instance.data)
        for exp in explanations:
            # Getting the perturbation matrices of all the explanations that are valid counterfactuals
            if self.oracle.predict(exp) != label:
                edge_freq_matrix = np.add(edge_freq_matrix, exp.data)

        norm_edge_freqs = edge_freq_matrix / np.max(edge_freq_matrix)
        
        embedded_features = { label:torch.from_numpy(instance.node_features) for label in range(self.dataset.num_classes) }
        edge_probabilities = { label:torch.from_numpy(norm_edge_freqs) for label in range(self.dataset.num_classes) }
        
        cf_candidate = self.sampler.sample(instance,
                                           self.oracle,
                                           embedded_features=embedded_features,
                                           edge_probabilities=edge_probabilities)
        
        return instance if not cf_candidate else cf_candidate
        