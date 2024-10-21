import copy
import random
import sys
import numpy as np
from src.core.explainer_base import Explainer
from src.dataset.instances.graph import GraphInstance
from src.explainer.future.metaheuristic.Tagging.simple_tagger import SimpleTagger
from typing import Generator

from src.explainer.future.metaheuristic.initial_solution_search.simple_searcher import SimpleSearcher
from src.explainer.future.metaheuristic.local_search.binary_model import BinaryModel
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.utils.cfg_utils import init_dflts_to_of
from src.utils.comparison import get_edge_differences
from src.utils.metrics.ged import GraphEditDistanceMetric

class LocalSearch(Explainer):
    def check_configuration(self):
        super().check_configuration()
        



    def init(self):
        super().init()
        self.neigh_factor = 8
        self.runtime_factor = 6
        
        self.tagger = SimpleTagger()
        
        self.searcher = SimpleSearcher()
        
        self.distance_metric = GraphEditDistanceMetric()  

    def explain(self, instance):
        self.G = instance
        self.N = instance.num_nodes
        self.E = instance.num_edges
        self.EPlus = int((self.N * (self.N-1)) / 2)
        
        self.M = BinaryModel(self.oracle, instance)
        
        self.labels = self.tagger.tag(instance)
        
        self.actual = None
        self.best = None

        result = self.get_approximation()
        
        minimal_explanation = LocalGraphCounterfactualExplanation(context=self.context,
                                                                    dataset=self.dataset,
                                                                    oracle=self.oracle,
                                                                    explainer=self,
                                                                    input_instance=instance,
                                                                    counterfactual_instances=[result])
        
        return minimal_explanation
        
        
    def get_approximation(self):
        
        print("Instance response: " + str(self.M.InitialResponse))
        # n = 0
        # print(self.labels)
        # for s in self.searcher.search(self.G, self.labels):
        #     n += 1
        #     print("initial attempt number " + str(n))
        #     print(s)
        #     (cf, new_g) = self.evaluate(s)
        #     if (cf):
        #         self.actual = s
        #         self.best = s
        #         print("Initial solution found after " + str(n) + " tries")
        #         break
        

        # Iterating over all the instances of the dataset
        
        min_ctf_dist = sys.float_info.max
        for ctf_candidate in self.dataset.instances:
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

        print("Initial solution: " + str(self.actual))
        print("Initial solution size: " + str(len(self.actual)))
        print("Initial solution response: ")
        (_, _) = self.evaluate(self.actual)
        
        # print(different_coords_list)
        
        n = self.runtime_factor * len(self.actual)
        while(n > 0):
            n-=1
            if(len(self.best) == 1) : break
            found = False
            
            print("Exploring removing neighborhood")
            for s in self.edge_remove(self.actual):
                (found_, _) = self.evaluate(s)
                if(found_ and len(s) < len(self.best)):
                    found = True
                    self.best = s
                    self.actual = s
                    n = self.runtime_factor * len(self.actual)
                    break
                
            if(found):
                print("Found solution of size: " + str(len(self.actual)))
                continue

            self.actual = self.reduce_random(self.best, int(len(self.actual) / 2))
            
            while(abs(len(self.best) - len(self.actual)) > 1):
                print("Exploring Swaping neighborhood")
                for s in self.edge_swap(self.actual):
                    (found_, _) = self.evaluate(s)
                    if(found_ and len(s) < len(self.best)):
                        found = True
                        self.best = s
                        self.actual = s
                        n = self.runtime_factor * len(self.actual)
                        break
                    
                if(found):
                    print("Found solution of size: " + str(len(self.actual)))
                    break

                self.actual = self.reduce_random(self.best, len(self.actual))
                
                print("Exploring Adding neighborhood")
                for s in self.edge_add(self.actual):
                    (found_, _) = self.evaluate(s)
                    if(found_ and len(s) < len(self.best)):
                        found = True
                        self.best = s
                        self.actual = s
                        n = self.runtime_factor * len(self.actual)
                        break
                    
                if(found):
                    print("Found solution of size: " + str(len(self.actual)))
                    break

                self.actual = self.reduce_random(self.best, int(len(self.actual) + (len(self.best) - len(self.actual))  / 2))
                
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
        for i in range(1, min(len(self.actual), (self.EPlus - len(self.actual))) + 1):
            for _ in range(self.neigh_factor ** 3):
                yield self.swap_random(set(solution.copy()), i)
                
    
    def edge_add(self, solution : set[int]) -> Generator[set[int], None, None]:
        for i in range(1, (len(self.best) - len(self.actual)) + 1):
            for _ in range(self.neigh_factor ** 3):
                yield self.add_random(set(solution.copy()), i)
                
                
    
    def edge_remove(self, solution : set[int]) -> Generator[set[int], None, None]:
        for i in range(len(self.actual), 0, -1):
            for _ in range(self.neigh_factor ** 3):
                yield self.remove_random(set(solution.copy()), i)
                
    
    
    
            
            
                