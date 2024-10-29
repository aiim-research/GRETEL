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

class LocalSearchMultiSolutions(LocalSearch):
    def check_configuration(self):
        super().check_configuration()
        
        if 'stack_limit' not in self.local_config['parameters']:
            self.local_config['parameters']['stack_limit'] = 10


    def init(self):
        super().init()
        
        self.stack_limit = self.local_config['parameters']['stack_limit']


    def get_approximation(self):
        
        self.logger.info("Instance response: " + str(self.M.InitialResponse))
        self.logger.info("Initial solution: " + str(self.actual))
        self.logger.info("Initial solution size: " + str(len(self.actual)))
        
        n = min(self.max_runtime, self.runtime_factor * len(self.actual))
        
        best_queue = [self.actual]
        
        while(len(best_queue) > 0):
            self.logger.info("queue len: " + str(len(best_queue)) + ", n: " + str(n))
            n-=1
            
            if(len(self.best) == 1) : break
            found = False
            self.actual = best_queue[0]
            
            for s in self.edge_remove(self.actual):
                (found_, _) = self.evaluate(s)
                if(found_ and len(s) < len(self.best)):
                    found = True
                    best_queue = [s]
                    self.best = s
                    self.actual = s
                    n = min(self.max_runtime, self.runtime_factor * len(self.actual))
                    
                    for s_ in self.edge_swap(self.actual):
                        (found__, _) = self.evaluate(s_)
                        if(found__):
                            best_queue.append(s_)
                            if(len(best_queue) > self.stack_limit):
                                break

                    break
                
            if(found):
                self.logger.info("============> Found solution with size: " + str(len(self.actual)))
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
                        if(not found):
                            best_queue = [s]
                            self.best = s
                            self.actual = s
                        else:
                            best_queue.append(s)
                            if(len(best_queue) > self.stack_limit):
                                break
                        found = True
                        n = min(self.max_runtime, self.runtime_factor * len(self.actual))
                    
                if(found):
                    self.logger.info("============> Found solution with size: " + str(len(self.actual)))
                    break

                self.actual = self.reduce_random(self.best, len(self.actual))
                self.logger.info("actual ===> " + str(len(self.actual)))
                
                for s in self.edge_add(self.actual):
                    (found_, _) = self.evaluate(s)
                    if(found_ and len(s) < len(self.best)):
                        found = True
                        best_queue = [s]
                        self.best = s
                        self.actual = s
                        n = min(self.max_runtime, self.runtime_factor * len(self.actual))
                        
                        for s_ in self.edge_swap(self.actual):
                            (found__, _) = self.evaluate(s_)
                            if(found__):
                                best_queue.append(s_)
                                if(len(best_queue) > self.stack_limit):
                                    break

                        break
                    
                if(found):
                    self.logger.info("============> Found solution with size: " + str(len(self.actual)))
                    break
                
                to_expand = int(((len(self.best) - len(self.actual)) / 2)) + 1
                expand = len(self.actual) + min(to_expand, random.randint(1, to_expand * 4))
                # self.logger.info("expand: " + str(expand) + ", best: " + str(len(self.best)))
                if(expand > len(self.best)): break
                self.actual = self.reduce_random(self.best, expand)
                self.logger.info("actual +++> " + str(len(self.actual)))
                
            if(n < 0):
                best_queue.pop(0)
                n = min(self.max_runtime, self.runtime_factor * len(self.actual))
                
        (_, result) = self.evaluate(self.best)
        return result
    
    
    
            
            
                