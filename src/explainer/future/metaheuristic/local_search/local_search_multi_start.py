import copy
import random
import sys
import numpy as np
from src.core.explainer_base import Explainer
from src.dataset.instances.base import DataInstance
from src.dataset.instances.graph import GraphInstance
from src.explainer.future.meta.minimizer.base import ExplanationMinimizer
from src.explainer.future.metaheuristic.Tagging.simple_tagger import SimpleTagger
from typing import Generator

from src.explainer.future.metaheuristic.initial_solution_search.simple_searcher import SimpleSearcher
from src.explainer.future.metaheuristic.local_search.binary_model import BinaryModel
from src.explainer.future.metaheuristic.local_search.local_search import LocalSearch
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.utils.cfg_utils import init_dflts_to_of
from src.utils.comparison import get_edge_differences
from src.utils.metrics.ged import GraphEditDistanceMetric

class LocalSearchMultiStart(LocalSearch):
    def check_configuration(self):
        super().check_configuration()
        
        if 'starting_amount' not in self.local_config['parameters']:
            self.local_config['parameters']['starting_amount'] = 10


    def init(self):
        super().init()
        
        self.starting_amount = self.local_config['parameters']['starting_amount']


    def minimize(self, explaination: LocalGraphCounterfactualExplanation) -> DataInstance:
        instance = explaination.input_instance
        self.G = instance
        self.N = instance.num_nodes
        self.E = instance.num_edges
        self.EPlus = int((self.N * (self.N-1)) / 2)
        
        self.M = BinaryModel(self.oracle, instance)
        
        self.labels = self.tagger.tag(instance)
        
        
        min_ctf = explaination.counterfactual_instances[0]
        min_ctf_dist = self.distance_metric.evaluate(self.G, min_ctf, self.oracle)
        n = 0
        self.logger.info("##################################")
        for ctf_candidate in explaination.counterfactual_instances:
            n+=1
            if(n>=self.starting_amount or min_ctf_dist < n):
                break
            candidate_label = self.oracle.predict(ctf_candidate)

            if self.M.InitialResponse != candidate_label:
                _, diff_matrix = get_edge_differences(self.G, ctf_candidate)
                different_coordinates = np.where(diff_matrix == 1)
                
                different_coords_list = list(zip(different_coordinates[0], different_coordinates[1]))
                self.actual = self.tagger.get_indices(self.labels, different_coords_list)
                self.best = self.actual
                
                self.logger.info("####> Exploring from initial solutionof size: " + str(len(self.actual)))

                ctf_minimized = self.get_approximation()
                
                ctf_distance = self.distance_metric.evaluate(self.G, ctf_minimized, self.oracle)
                
                if ctf_distance < min_ctf_dist:
                    min_ctf_dist = ctf_distance
                    min_ctf = ctf_minimized
                    self.logger.info("####> Found upgrade with size: " + str(min_ctf_dist))
        self.logger.info("##################################")
        
        return min_ctf

    
    
    
            
            
                