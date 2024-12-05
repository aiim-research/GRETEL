import copy
import math
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
from src.explainer.future.metaheuristic.local_search.cache import FixedSizeCache
from src.explainer.future.metaheuristic.manipulation.methods import average_smoothing, feature_aggregation, heat_kernel_diffusion, laplacian_regularization, random_walk_diffusion, weighted_smoothing
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.utils.cfg_utils import init_dflts_to_of
from src.utils.comparison import get_edge_differences
from src.utils.metrics.ged import GraphEditDistanceMetric
from collections import OrderedDict

class LocalSearch(ExplanationMinimizer):
    def check_configuration(self):
        super().check_configuration()
        
        if 'neigh_factor' not in self.local_config['parameters']:
            self.local_config['parameters']['neigh_factor'] = 4
        
        if 'runtime_factor' not in self.local_config['parameters']:
            self.local_config['parameters']['runtime_factor'] = 4
        
        if 'max_runtime' not in self.local_config['parameters']:
            self.local_config['parameters']['max_runtime'] = 50
            
        if 'max_neigh' not in self.local_config['parameters']:
            self.local_config['parameters']['max_neigh'] = 30
            
        if 'attributed' not in self.local_config['parameters']:
            self.local_config['parameters']['attributed'] = False
            
        if 'max_oracle_calls' not in self.local_config['parameters']:
            self.local_config['parameters']['max_oracle_calls'] = 10000
        



    def init(self):
        super().init()
        self.logger = self.context.logger
        self.neigh_factor = self.local_config['parameters']['neigh_factor']
        self.runtime_factor = self.local_config['parameters']['runtime_factor']
        self.max_runtime = self.local_config['parameters']['max_runtime']
        self.max_neigh = self.local_config['parameters']['max_neigh']
        self.attributed = self.local_config['parameters']['attributed']
        self.max_oracle_calls = self.local_config['parameters']['max_oracle_calls']
        
        self.tagger = SimpleTagger()
        
        self.searcher = SimpleSearcher()
        
        self.distance_metric = GraphEditDistanceMetric()  
        
        self.methods = [
            lambda data, features: average_smoothing(data, features, iterations=1),
            lambda data, features: weighted_smoothing(data, features, iterations=1),
            lambda data, features: laplacian_regularization(data, features, lambda_reg=0.01, iterations=1),
            lambda data, features: feature_aggregation(data, features, alpha=0.5, iterations=1),
            lambda data, features: heat_kernel_diffusion(data, features, t=0.5),
            lambda data, features: random_walk_diffusion(data, features, steps=1)
        ]
        
        

    def minimize(self, explaination: LocalGraphCounterfactualExplanation) -> DataInstance:
        print("-------------")
        instance = explaination.input_instance
        self.G = instance
        self.N = instance.num_nodes
        self.E = instance.num_edges
        self.EPlus = int((self.N * (self.N-1)) / 2)
        
        self.M = BinaryModel(self.oracle, instance)
        
        self.labels = self.tagger.tag(instance)
        
        
        min_ctf = explaination.counterfactual_instances[0]
        # min_ctf_dist = self.distance_metric.evaluate(self.G, min_ctf, self.oracle)
        # for ctf_candidate in explaination.counterfactual_instances:
        #     candidate_label = self.oracle.predict(ctf_candidate)

        #     if self.M.InitialResponse != candidate_label:
        #         ctf_distance = self.distance_metric.evaluate(self.G, ctf_candidate, self.oracle)
                
        #         if ctf_distance < min_ctf_dist:
        #             min_ctf_dist = ctf_distance
        #             min_ctf = ctf_candidate
        
        _, diff_matrix = get_edge_differences(self.G, min_ctf)
        different_coordinates = np.where(diff_matrix == 1)        
        different_coords_list = list(zip(different_coordinates[0], different_coordinates[1]))
        # Filter to avoid duplicate edges in undirected graphs
        filtered_coords_list = [coord for coord in different_coords_list if coord[0] < coord[1]]
        actual = self.tagger.get_indices(self.labels, filtered_coords_list)
        
        best = actual
        

        
        if(len(actual) == 0):
            return min_ctf
        
        self.cache = FixedSizeCache(capacity=500000)
        result = self.get_approximation(actual, best, min_ctf)
        
        # candidate_label = self.oracle.predict(result)
        # if self.M.InitialResponse == candidate_label:
        #     result = self.get_evaluation([])
        #     self.logger.info("Contrafractual no encontrado")
        
        return result
        
        
    def get_approximation(self, actual, best, min_ctf):
        self.logger.info("Initial solution: " + str(actual))
        self.logger.info("Initial solution size: " + str(len(actual)))

        result = min_ctf
        
        n = min(self.max_runtime, self.runtime_factor * len(actual))
        self.k = 0
        while(n > 0):
            self.logger.info("n: " + str(n))
            self.logger.info("k: " + str(self.k))
            n-=1
            if(len(best) == 1) : break
            if(self.k > self.max_oracle_calls) :
                 self.logger.info("Oracle calls limit reached")
                 break
            found = False
            actual = best
            # self.logger.info("actual ---> " + str(len(actual)))
            
            for s in self.edge_remove(actual):
                if(self.cache.contains(s)):
                    continue
                self.cache.add(s)
                found_, inst = self.evaluate(s)
                if(found_ and len(s) < len(best)):
                    found = True
                    best = s
                    actual = s
                    result = inst
                    n = min(self.max_runtime, self.runtime_factor * len(actual))
                    break
                
            if(found):
                self.logger.info("============> (-) Found solution with size: " + str(len(actual)))
                continue
            
            half = int(len(actual) / 2)
            reduce = min(half, random.randint(1, half * 4))
            actual = self.reduce_random(best, reduce)
            self.logger.info("actual ---> " + str(len(actual)))
            
            while(len(best) - len(actual) > 1):
                n-=1
                for s in self.edge_swap(actual):
                    if(self.cache.contains(s)):
                        # print("in cache")
                        continue
                        
                    self.cache.add(s)
                    found_, inst = self.evaluate(s)
                    if(found_ and len(s) < len(best)):
                        found = True
                        best = s
                        actual = s
                        result = inst
                        n = min(self.max_runtime, self.runtime_factor * len(actual))
                        break
                    
                if(found):
                    self.logger.info("============> (=) Found solution with size: " + str(len(actual)))
                    break

                actual = self.reduce_random(best, len(actual))
                self.logger.info("actual ===> " + str(len(actual)))
                
                for s in self.edge_add(actual, best):
                    if(self.cache.contains(s)):
                        # print("in cache")
                        continue
                        
                    self.cache.add(s)
                    found_, inst = self.evaluate(s)
                    if(found_ and len(s) < len(best)):
                        found = True
                        best = s
                        actual = s
                        result = inst
                        n = min(self.max_runtime, self.runtime_factor * len(actual))
                        break
                    
                if(found):
                    self.logger.info("============> (+) Found solution with size: " + str(len(actual)))
                    break
                
                to_expand = int(((len(best) - len(actual)) / 2)) + 1
                expand = len(actual) + min(to_expand, random.randint(1, to_expand * 4))
                # self.logger.info("expand: " + str(expand) + ", best: " + str(len(best)))
                if(expand > len(best)): break
                actual = self.reduce_random(best, expand)
                self.logger.info("actual +++> " + str(len(actual)))
          
        if(self.oracle.predict(result) == self.oracle.predict(self.G)):
            self.logger.info("ERROR, returning non ctf ")
            self.logger.info("instance -> " + str(self.oracle.predict(self.G)))
            self.logger.info("result -> " + str(self.oracle.predict(result)))
        return result
    
    def evaluate(self, solution : set[int]) -> tuple[bool, GraphInstance]:
        new_data = np.copy(self.G.data)
        self.disturb(new_data, self.G.directed, solution)
        
        # If the dataset has attributes in the nodes, then lets explore those with the methods
        if(self.attributed):
            for method in self.methods:
                self.k += 1
                node_features = method(new_data, self.G.node_features)
                new_g = GraphInstance(id=self.G.id,
                                        label=0,
                                        data=new_data,
                                        directed=self.G.directed,
                                        node_features= node_features)
                if(self.M.classify(new_g)): return (True, new_g)
        
        # If the dataset does not has attributes, then it has ficticial attributes for GCN to work,
        # in that case, we just call the manipulator method
        else:
            self.k += 1
            new_g = GraphInstance(id=self.G.id,
                                        label=0,
                                        data=new_data,
                                        directed=self.G.directed,
                                        node_features= self.G.node_features)
            self.dataset.manipulate(new_g)
            if(self.M.classify(new_g)): return (True, new_g)

        return (False, None)
    
        
    def disturb(self, data, directed, solution : set[int]):
        for i in solution:
            (n1, n2) = self.labels[i]
            data[n1, n2] = (data[n1, n2] + 1) % 2
            if(not directed):
                data[n2, n1] = (data[n2, n1] + 1) % 2


    def swap_random(self, solution : set[int], i: int):  
        self.remove_random(solution, i)
        self.add_random(solution, i)
        
        return solution
    
    def add_random(self, solution : set[int], i: int):
        available_numbers = set(range(1, self.EPlus)) - solution
        
        if len(available_numbers) < i:
            raise ValueError("Not enough available numbers to add.")
        
        numbers_to_add = random.sample(available_numbers, i)
        
        solution.update(numbers_to_add)
        
        return solution
    
    def remove_random(self, solution : set[int], i: int):
        numbers_to_remove = random.sample(solution, i)
        
        solution.difference_update(numbers_to_remove)
        
        return solution
    
    def reduce_random(self, solution : set[int], i: int):
        if len(solution) < i:
            raise ValueError("The set does not have enough elements.")
        
        selected_elements = set(random.sample(solution, i))
        
        return selected_elements


    def edge_swap(self, solution : set[int]) -> Generator[set[int], None, None]:
        cealing = min(len(solution), (self.EPlus - len(solution))) + 1
        step = int(cealing / self.max_neigh) + 1
        for i in range(1, cealing, step):
            for _ in range(self.neigh_factor ** 2):
                yield self.swap_random(set(solution.copy()), i)
                
    
    def edge_add(self, solution : set[int], best) -> Generator[set[int], None, None]:
        cealing = (len(best) - len(solution)) + 1
        step = int(cealing / self.max_neigh) + 1
        for i in range(1, cealing, step):
            for _ in range(self.neigh_factor ** 2):
                yield self.add_random(set(solution.copy()), i)
                
                
    
    def edge_remove(self, solution : set[int]) -> Generator[set[int], None, None]:
        cealing = len(solution)
        step = int((cealing / self.max_neigh) + 1) 
        # cealing = random.randint(cealing - step, cealing)
        for i in range(0, cealing, step):
            for _ in range(self.neigh_factor ** 3):
                yield self.remove_random(set(solution.copy()), i)
                
    def write(self):
        pass

    def read(self):
        pass
                
    
    
    
            
            
                