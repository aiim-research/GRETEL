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
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.utils.cfg_utils import init_dflts_to_of
from src.utils.comparison import get_edge_differences
from src.utils.metrics.ged import GraphEditDistanceMetric

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
        



    def init(self):
        super().init()
        self.logger = self.context.logger
        self.neigh_factor = self.local_config['parameters']['neigh_factor']
        self.runtime_factor = self.local_config['parameters']['runtime_factor']
        self.max_runtime = self.local_config['parameters']['max_runtime']
        self.max_neigh = self.local_config['parameters']['max_neigh']
        
        self.tagger = SimpleTagger()
        
        self.searcher = SimpleSearcher()
        
        self.distance_metric = GraphEditDistanceMetric()  

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
        for ctf_candidate in explaination.counterfactual_instances:
            candidate_label = self.oracle.predict(ctf_candidate)

            if self.M.InitialResponse != candidate_label:
                ctf_distance = self.distance_metric.evaluate(self.G, ctf_candidate, self.oracle)
                
                if ctf_distance < min_ctf_dist:
                    min_ctf_dist = ctf_distance
                    min_ctf = ctf_candidate
        
        _, diff_matrix = get_edge_differences(self.G, min_ctf)
        different_coordinates = np.where(diff_matrix == 1)        
        different_coords_list = list(zip(different_coordinates[0], different_coordinates[1]))
        self.actual = self.tagger.get_indices(self.labels, different_coords_list)
        
        self.best = self.actual
        
        if(len(self.actual) == 0):
            (_, result) = self.evaluate(self.best)
            return result
        
        # if(len(self.actual) == 0):
        #     self.logger.info("Tomando inicial random")
        #     self.actual = next(self.searcher.search(self.G, self.labels, int(self.N)), None)
        
        result = self.get_approximation()
        
        # candidate_label = self.oracle.predict(result)
        # if self.M.InitialResponse == candidate_label:
        #     (_, result) = self.evaluate([])
        #     self.logger.info("Contrafractual no encontrado")
        
        return result
        
        
    def get_approximation(self):
        
        self.logger.info("Instance response: " + str(self.M.InitialResponse))
        self.logger.info("Initial solution: " + str(self.actual))
        self.logger.info("Initial solution size: " + str(len(self.actual)))
        
        # self.logger.info(different_coords_list)
        
        n = min(self.max_runtime, self.runtime_factor * len(self.actual))
        while(n > 0):
            self.logger.info("n: " + str(n))
            n-=1
            if(len(self.best) == 1) : break
            found = False
            self.actual = self.best
            # self.logger.info("actual ---> " + str(len(self.actual)))
            
            for s in self.edge_remove(self.actual):
                (found_, _) = self.evaluate(s)
                if(found_ and len(s) < len(self.best)):
                    found = True
                    self.best = s
                    self.actual = s
                    n = min(self.max_runtime, self.runtime_factor * len(self.actual))
                    break
                
            if(found):
                self.logger.info("============> (-) Found solution with size: " + str(len(self.actual)))
                continue
            
            half = int(len(self.actual) / 2)
            reduce = min(half, random.randint(1, half * 4))
            self.actual = self.reduce_random(self.best, reduce)
            self.logger.info("actual ---> " + str(len(self.actual)))
            
            while(len(self.best) - len(self.actual) > 1):
                n-=1
                for s in self.edge_swap(self.actual):
                    (found_, _) = self.evaluate(s)
                    if(found_ and len(s) < len(self.best)):
                        found = True
                        self.best = s
                        self.actual = s
                        n = min(self.max_runtime, self.runtime_factor * len(self.actual))
                        break
                    
                if(found):
                    self.logger.info("============> (=) Found solution with size: " + str(len(self.actual)))
                    break

                self.actual = self.reduce_random(self.best, len(self.actual))
                self.logger.info("actual ===> " + str(len(self.actual)))
                
                for s in self.edge_add(self.actual):
                    (found_, _) = self.evaluate(s)
                    if(found_ and len(s) < len(self.best)):
                        found = True
                        self.best = s
                        self.actual = s
                        n = min(self.max_runtime, self.runtime_factor * len(self.actual))
                        break
                    
                if(found):
                    self.logger.info("============> (+) Found solution with size: " + str(len(self.actual)))
                    break
                
                to_expand = int(((len(self.best) - len(self.actual)) / 2)) + 1
                expand = len(self.actual) + min(to_expand, random.randint(1, to_expand * 4))
                # self.logger.info("expand: " + str(expand) + ", best: " + str(len(self.best)))
                if(expand > len(self.best)): break
                self.actual = self.reduce_random(self.best, expand)
                self.logger.info("actual +++> " + str(len(self.actual)))
                
        (_, result) = self.evaluate(self.best)
        return result
    
    
    def evaluate(self, solution : set[int]) -> tuple[bool, GraphInstance]:
        new_data = copy.deepcopy(self.G.data)
        self.disturb(new_data, self.G.directed, solution)
        
        new_g = GraphInstance(id=self.G.id,
                                label=0,
                                data=new_data,
                                node_features=self.G.node_features)
        result = self.M.classify(new_g)

        return (result, new_g)
        
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
        cealing = min(len(self.actual), (self.EPlus - len(self.actual))) + 1
        step = int(cealing / self.max_neigh) + 1
        for i in range(1, cealing, step):
            for _ in range(self.neigh_factor ** 3):
                yield self.swap_random(set(solution.copy()), i)
                
    
    def edge_add(self, solution : set[int]) -> Generator[set[int], None, None]:
        cealing = (len(self.best) - len(self.actual)) + 1
        step = int(cealing / self.max_neigh) + 1
        for i in range(1, cealing, step):
            for _ in range(self.neigh_factor ** 3):
                yield self.add_random(set(solution.copy()), i)
                
                
    
    def edge_remove(self, solution : set[int]) -> Generator[set[int], None, None]:
        floor = len(self.actual)
        step = int((floor / self.max_neigh) + 1) * -1
        for i in range(floor, 0, step):
            for _ in range(self.neigh_factor ** 3):
                yield self.remove_random(set(solution.copy()), i)
                
    def write(self):
        pass

    def read(self):
        pass
                
    
    
    
            
            
                