import copy
from typing import List
import numpy as np
import torch

from src.dataset.instances.graph import GraphInstance
from src.explainer.future.ensemble.aggregators.base import ExplanationAggregator
from src.utils.samplers.abstract_sampler import Sampler
from src.core.factory_base import get_instance_kvargs
from src.utils.cfg_utils import init_dflts_to_of
from src.utils.utils import pad_adj_matrix
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
import src.utils.explanations.functions as exp_tools


class ExplanationStochasticAggregator(ExplanationAggregator):

    def check_configuration(self):
        super().check_configuration()
        self.logger= self.context.logger

        dst_metric = 'src.utils.metrics.ged.GraphEditDistanceMetric'
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
        

    def real_aggregate(self, explanations: List[LocalGraphCounterfactualExplanation]) -> LocalGraphCounterfactualExplanation:
        # Extract the original instance and the counterfactual instances from the explanations
        input_inst = explanations[0].input_instance
        cf_instances = exp_tools.unpack_cf_instances(explanations)
        # Get the number of nodes of the bigger explanation instance
        max_dim = max(input_inst.data.shape[0], max([cf.data.shape[0] for cf in cf_instances]))

        # Adding the edges of all the counterfactual instances
        edge_freq_matrix = np.zeros((max_dim, max_dim))
        for cf in cf_instances:
            edge_freq_matrix[:cf.data.shape[0], :cf.data.shape[0]] += cf.data

        # Normalizing the frequency of appearance of each edge
        norm_edge_freqs = edge_freq_matrix / np.max(edge_freq_matrix)

        # Check that there are edges in the combined graph
        if np.any(edge_freq_matrix):
            embedded_features = { label:torch.from_numpy(input_inst.node_features) for label in range(self.dataset.num_classes) }
            edge_probabilities = { label:torch.from_numpy(norm_edge_freqs) for label in range(self.dataset.num_classes) }
            
            aggregated_instance = self.sampler.sample(input_inst, self.oracle,
                                               embedded_features=embedded_features,
                                               edge_probabilities=edge_probabilities)
            
            if aggregated_instance:
                self.dataset.manipulate(aggregated_instance)
                aggregated_instance.label = self.oracle.predict(aggregated_instance)
            
                aggregated_explanation = LocalGraphCounterfactualExplanation(context=self.context,
                                                                    dataset=self.dataset,
                                                                    oracle=self.oracle,
                                                                    explainer=None, # Will be added later by the ensemble
                                                                    input_instance=input_inst,
                                                                    counterfactual_instances=[aggregated_instance])

                return aggregated_explanation
        
        # The default behavior if an explanation was not produced is to return the input instance as explanation
        no_explanation = LocalGraphCounterfactualExplanation(context=self.context,
                                                            dataset=self.dataset,
                                                            oracle=self.oracle,
                                                            explainer=None, # Will be added later by the ensemble
                                                            input_instance=input_inst,
                                                            counterfactual_instances=[copy.deepcopy(input_inst)])
        return no_explanation
        