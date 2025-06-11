import copy
import math
import random
import sys
import numpy as np
from src.core.explainer_base import Explainer
from src.core.trainable_base import Trainable
from src.dataset.instances.base import DataInstance
from src.dataset.instances.graph import GraphInstance
from src.explainer.future.meta.minimizer.base import ExplanationMinimizer
from typing import Generator

from src.explainer.future.metaheuristic.initial_solution_search.simple_searcher import SimpleSearcher
from src.explainer.future.metaheuristic.local_search.binary_model import BinaryModel
from src.explainer.future.metaheuristic.local_search.cache import FixedSizeCache
from src.explainer.future.metaheuristic.manipulation.methods import average_smoothing, average_smoothing_zero, feature_aggregation, heat_kernel_diffusion, laplacian_regularization, random_walk_diffusion, weighted_smoothing, identity
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.utils.cfg_utils import init_dflts_to_of
from src.utils.comparison import get_edge_differences
from src.utils.metrics.ged import GraphEditDistanceMetric
from collections import OrderedDict

class LocalSearchTrainable(ExplanationMinimizer, Explainer, Trainable):
    def check_configuration(self):
        super().check_configuration()
        
        if 'neigh_factor' not in self.local_config['parameters']:
            self.local_config['parameters']['neigh_factor'] = 4
        
        if 'runtime_factor' not in self.local_config['parameters']:
            self.local_config['parameters']['runtime_factor'] = 4
        
        if 'max_runtime' not in self.local_config['parameters']:
            self.local_config['parameters']['max_runtime'] = 50
            
        if 'proportion' not in self.local_config['parameters']:
            self.local_config['parameters']['proportion'] = 25

        if 'max_neigh' not in self.local_config['parameters']:
            self.local_config['parameters']['max_neigh'] = 30
            
        if 'attributed' not in self.local_config['parameters']:
            self.local_config['parameters']['attributed'] = False
            
        if 'max_oracle_calls' not in self.local_config['parameters']:
            self.local_config['parameters']['max_oracle_calls'] = 10000
            
        if 'tagger' not in self.local_config['parameters']:
            self.local_config['parameters']['tagger'] = "src.explainer.future.metaheuristic.Tagging.simple_tagger.SimpleTagger"
        

    def get_class_from_string(self ,class_path):
        module_path, class_name = class_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)

    def convert(self, method):
        if method == "average_smoothing":
            return lambda data, features: average_smoothing(data, features, iterations=1)
        elif method == "average_smoothing_zero":
            return lambda data, features: average_smoothing_zero(data, features, iterations=1)
        elif method == "weighted_smoothing":
            return lambda data, features: weighted_smoothing(data, features, iterations=1)
        elif method == "laplacian_regularization":
            return lambda data, features: laplacian_regularization(data, features, lambda_reg=0.01, iterations=1)
        elif method == "feature_aggregation":
            return lambda data, features: feature_aggregation(data, features, alpha=0.5, iterations=1)
        elif method == "heat_kernel_diffusion":
            return lambda data, features: heat_kernel_diffusion(data, features, t=0.5)
        elif method == "random_walk_diffusion":
            return lambda data, features: random_walk_diffusion(data, features, steps=1)
        elif method == "identity":
            return lambda data, features: identity(data, features)

    def init(self):
        super().init()
        self.training = False
        self.last_method = -1
        self.logger = self.context.logger
        self.neigh_factor = self.local_config['parameters']['neigh_factor']
        self.runtime_factor = self.local_config['parameters']['runtime_factor']
        self.max_runtime = self.local_config['parameters']['max_runtime']
        self.max_neigh = self.local_config['parameters']['max_neigh']
        self.attributed = self.local_config['parameters']['attributed']
        self.max_oracle_calls = self.local_config['parameters']['max_oracle_calls']
        self.proportion = self.local_config['parameters']['proportion']
        
        tagger_direction = self.local_config['parameters']['tagger']
        self.tagger = self.get_class_from_string(tagger_direction)()
        
        self.searcher = SimpleSearcher()
        
        self.distance_metric = GraphEditDistanceMetric() 
        self.device = "cpu" 
        
        self.model = {}
    def minimize(self, explaination: LocalGraphCounterfactualExplanation) -> DataInstance:
        #self.logger.info("The firsts " + str(self.last_method) + " are in using")

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
        
        if(self.oracle.predict(min_ctf) == self.oracle.predict(self.G)):
            #self.logger.info("MIN_CTF label equal to G label")
            # find an auxiliar solution 
            min_ctf = self.explain(self.G).counterfactual_instances[0]
            _, diff_matrix = get_edge_differences(self.G, min_ctf)
            different_coordinates = np.where(diff_matrix == 1)        
            different_coords_list = list(zip(different_coordinates[0], different_coordinates[1]))
            # Filter to avoid duplicate edges in undirected graphs
            filtered_coords_list = [coord for coord in different_coords_list if coord[0] < coord[1]]
            actual = self.tagger.get_indices(self.labels, filtered_coords_list)
            best = actual
        
        self.cache = FixedSizeCache(capacity=500000)
        result = self.get_approximation(actual, best, min_ctf)
        
        # candidate_label = self.oracle.predict(result)
        # if self.M.InitialResponse == candidate_label:
        #     result = self.get_evaluation([])
        #     self.logger.info("Contrafractual no encontrado")
        
        return result
        
        
    def get_approximation(self, actual, best, min_ctf):
        #self.logger.info("Initial solution: " + str(actual))
        self.logger.info("Initial solution size: " + str(len(actual)))

        result = min_ctf
        
        n = min(self.max_runtime, self.runtime_factor * len(actual))
        self.k = 0
        while(n > 0):
            #self.logger.info("n: " + str(n))
            #self.logger.info("k: " + str(self.k))
            n-=1
            if(len(best) == 1) : break
            if(self.k > self.max_oracle_calls) :
                 #self.logger.info("Oracle calls limit reached")
                 break
            found = False
            actual = best
            #self.logger.info("actual ---> " + str(len(actual)))
            
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
                #self.logger.info("============> (-) Found solution with size: " + str(len(actual)))
                continue
            
            half = int(len(actual) / 2)
            reduce = min(half, random.randint(1, half * 4))
            actual = self.reduce_random(best, reduce)
            #self.logger.info("actual ---> " + str(len(actual)))
            
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
                    #self.logger.info("============> (=) Found solution with size: " + str(len(actual)))
                    break

                actual = self.reduce_random(best, len(actual))
                #self.logger.info("actual ===> " + str(len(actual)))
                
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
                    #self.logger.info("============> (+) Found solution with size: " + str(len(actual)))
                    break
                
                to_expand = int(((len(best) - len(actual)) / 2)) + 1
                expand = len(actual) + min(to_expand, random.randint(1, to_expand * 4))
                # self.logger.info("expand: " + str(expand) + ", best: " + str(len(best)))
                if(expand > len(best)): break
                actual = self.reduce_random(best, expand)
                #self.logger.info("actual +++> " + str(len(actual)))
          
        if(self.oracle.predict(result) == self.oracle.predict(self.G)):
            self.logger.info("ERROR, returning non ctf ")
            self.logger.info("instance -> " + str(self.oracle.predict(self.G)))
            self.logger.info("result -> " + str(self.oracle.predict(result)))
        self.logger.info("final solution size: " + str(len(actual)))
        return result
    
    def evaluate(self, solution : set[int]) -> tuple[bool, GraphInstance]:
        new_data = np.copy(self.G.data)
        self.disturb(new_data, self.G.directed, solution)
        
        # If the dataset has attributes in the nodes, then lets explore those with the methods
        
        if(self.attributed):
            if self.training:
                ans = None
                for i, (score, method) in enumerate(self.model["methods"]):
                    true_method = self.convert(method)
                    self.k += 1
                    node_features = true_method(new_data, self.G.node_features)
                    new_g = GraphInstance(id=self.G.id,
                                        label=0,
                                        data=new_data,
                                        directed=self.G.directed,
                                        node_features=node_features)
                    
                    if self.M.classify(new_g): 
                        ans = (True, new_g)
                        self.model["methods"][i] = (score + 1, method)
                    else:
                        self.model["methods"][i] = (score - 1, method)

                if ans is not None:
                    return ans
            else:
                for i, (score, method) in enumerate(self.model["methods"]):
                    true_method = self.convert(method)
                    self.last_method = max(self.last_method, i+1)
                    self.k += 1
                    node_features = true_method(new_data, self.G.node_features)
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
                yield self.tagger.swap(set(solution.copy()), i)
                
    def edge_add(self, solution : set[int], best) -> Generator[set[int], None, None]:
        cealing = (len(best) - len(solution)) + 1
        step = int(cealing / self.max_neigh) + 1
        for i in range(1, cealing, step):
            for _ in range(self.neigh_factor ** 2):
                yield self.tagger.add(set(solution.copy()), i)
                  
    def edge_remove(self, solution : set[int]) -> Generator[set[int], None, None]:
        cealing = len(solution)
        step = int((cealing / self.max_neigh) + 1) 
        # cealing = random.randint(cealing - step, cealing)
        for i in range(0, cealing, step):
            for _ in range(self.neigh_factor ** 3):
                yield self.tagger.remove(set(solution.copy()), i)
                
    def real_fit(self):
        super().real_fit()

    def fit(self):
        self.logger.info("start training")
        self.training = True
        self.train_medoid()
        self.train_methods()
        self.training = False
        self.train_parameters()
        self.logger.info("end training")
        super().fit()

    def train_medoid(self):
        self.logger.info("start train_medoid")
        # Get the category of the graphs
        categorized_graph = [(self.oracle.predict(graph), graph) for graph in self.dataset.instances]
        
        # Groups the graph by category
        graphs_by_category = {}
        for category, graph in categorized_graph:
            if category not in graphs_by_category:
                graphs_by_category[category] = []
            graphs_by_category[category].append(graph)
        
        # Get the medoid of each category
        medoids = {}
        for category, graphs in graphs_by_category.items():
            graphs_distance_total = []
            
            for graph in graphs:
                distance = 0
                
                for category_, graphs_ in graphs_by_category.items():
                    if category == category_:
                        continue
                    for graph_ in graphs_: 
                        distance += self.distance_metric.evaluate(graph, graph_)
                
                graphs_distance_total.append((graph, distance))
            
            min_distance = float('inf')
            medoid = None
            
            for graph, distance in graphs_distance_total:
                if min_distance > distance:
                    min_distance = distance
                    medoid = graph
            
            medoids[category] = medoid
        self.model["medoids"] = medoids
        self.logger.info("end train_medoid")

    def train_methods(self):
        self.logger.info("start train_methods")
        methods = [
            "average_smoothing",
            "average_smoothing_zero",
            "weighted_smoothing",
            "laplacian_regularization",
            "feature_aggregation",
            "heat_kernel_diffusion",
            "random_walk_diffusion",
            "identity"
        ]

        self.model["methods"] = [(0, method) for method in methods]
        
        for instance in random.sample(self.dataset.instances, k=len(self.dataset.instances)):  
            self.logger.info("new instance")
            exp = self.explain(instance=instance)
            self.minimize(exp)
        
        self.model["methods"] = sorted(self.model["methods"], key=lambda x: x[0], reverse=True) 
        mid = (self.model["methods"][0][0] + self.model["methods"][7][0]) // 2
        self.model["methods"] = list(filter(lambda x: x[0] >= mid, self.model["methods"]))
        for i, (score, method) in enumerate(self.model["methods"]):
            self.logger.info(f"Score: {score}, Method: {method}")

        self.logger.info("end train_methods")
   
    def explain(self, instance):
        # Get the category of the instance
        category = self.oracle.predict(instance)
        
        # Get the closest medoid to the instance that belong to a different category 
        min_distance = float('inf')
        closest_medoid = None
        for other_category, medoid in self.model["medoids"].items():
            if other_category != category:
                distance = self.distance_metric.evaluate(instance, medoid)
                if distance < min_distance:
                    min_distance = distance
                    closest_medoid = medoid       

        # Create a graph's instance of the closest medoid
        cf_instance = GraphInstance(id=closest_medoid.id, label=closest_medoid.label, data=closest_medoid.data, node_features=closest_medoid.node_features)

        exp = LocalGraphCounterfactualExplanation(context=self.context, dataset=self.dataset, oracle=self.oracle, explainer=self, input_instance=instance, counterfactual_instances=[cf_instance])

        return exp  

    def perturb_neigh(self, neigh_factor):
        delta = int(round(np.random.normal(0, 5)))
        new_factor = neigh_factor + delta
        return min(max(1, new_factor), 15)
        
    def perturb_runtime(self, runtime_factor):
        delta = int(round(np.random.normal(0, 5)))
        new_factor = runtime_factor + delta
        return min(max(1, new_factor), 15)
        
    def perturb_max_runtime(self, max_runtime):
        delta = int(round(np.random.normal(0, 35)))
        new_runtime = max_runtime + delta
        return min(max(1, new_runtime), 80)
    
    def perturb_max_neigh(self, max_neigh):
        delta = int(round(np.random.normal(0, 25)))
        new_max_neigh = max_neigh + delta
        return min(max(1, new_max_neigh), 80)
    
    #def perturb_max_oracle(self, max_oracle_calls):
    #    delta = int(round(np.random.normal(0, 500)))
    #    new_max_oracle = max_oracle_calls + delta
    #    return min(max(1, new_max_oracle), 15000)
    
    def perturb_parameter(self, parameters):
        new_params = {
            'neigh_factor': self.perturb_neigh(parameters['neigh_factor']),
            'runtime_factor': self.perturb_runtime(parameters['runtime_factor']),
            'max_runtime': self.perturb_max_runtime(parameters['max_runtime']),
            'max_neigh': self.perturb_max_neigh(parameters['max_neigh']),
            #'max_oracle_calls': self.perturb_max_oracle(parameters['max_oracle_calls'])
        }
        return new_params
    
    def merge_parameters(self, parameters_a, parameters_b):
        new_params = {
            'neigh_factor': (parameters_a['neigh_factor'] + parameters_b['neigh_factor']) // 2,
            'runtime_factor': (parameters_a['runtime_factor'] + parameters_b['runtime_factor']) // 2,
            'max_runtime': (parameters_a['max_runtime'] + parameters_b['max_runtime']) // 2,
            'max_neigh': (parameters_a['max_neigh'] + parameters_b['max_neigh']) // 2
            #'max_oracle_calls': (parameters_a['max_oracle_calls'] + parameters_b['max_oracle_calls']) // 2
        }
        return new_params
    
    def try_parameters(self, parameters):
        self.neigh_factor = parameters['neigh_factor']
        self.runtime_factor = parameters['runtime_factor']
        self.max_runtime = parameters['max_runtime']
        self.max_neigh = parameters['max_neigh']
        #self.max_oracle_calls = parameters['max_oracle_calls']

    def train_parameters(self):
        self.logger.info("start train_parameters")
        base = {
            'neigh_factor': self.neigh_factor,
            'runtime_factor': self.runtime_factor,
            'max_runtime': self.max_runtime,
            'max_neigh': self.max_neigh
            #'max_oracle_calls': self.max_oracle_calls
        }
        
        self.logger.info("Generating candidates")
        candidates = []
        candidates.append({'val': 0,'oracle_calls': 0,'params': base})
        for _ in range(15):
            candidate = {
                'val': 0,
                'oracle_calls': 0,
                'params': self.perturb_parameter(base)
            }
            base = candidate['params']
            candidates.append(candidate)
        
        epochs = 0

        while epochs < 20:
        
            epochs+=1
            self.logger.info("Epoch: " + str(epochs))

            sample_instances = random.sample(self.dataset.instances, 
                                         k=len(self.dataset.instances)//self.proportion)
            
            for candidate in candidates:
                candidate['val'] = 0
                candidate['oracle_calls'] = 0
        
            for instance in sample_instances:
                for candidate in candidates:
                    self.try_parameters(candidate['params'])
                    exp = self.explain(instance=instance)
                    solution = self.minimize(exp)
                    candidate['val'] += get_edge_differences(instance, solution)[0]
                    candidate['oracle_calls'] += self.k
        
            self.logger.info("Selecting candidates")
            candidates.sort(key=lambda x: (x['val'], x['oracle_calls']))
            candidates = candidates[:8]
            random.shuffle(candidates)
            best_candidates = []
        
            self.logger.info("Merging candidates")
            while candidates:
                a = candidates.pop()
                b = candidates.pop()

                merged = self.merge_parameters(a['params'], b['params'])
                best_candidates.append({'val': 0, 'oracle_calls': 0 ,'params': merged})

                mutated = self.perturb_parameter(a['params'])
                best_candidates.append({'val': 0, 'oracle_calls':0, 'params': mutated})

                mutated = self.perturb_parameter(b['params'])
                best_candidates.append({'val': 0, 'oracle_calls':0, 'params': mutated})

                if (a['val'], a['oracle_calls']) <=  (b['val'], b['oracle_calls']):
                    best_candidates.append({'val': 0, 'oracle_calls': 0 ,'params': a['params']})
                else:
                    best_candidates.append({'val': 0, 'oracle_calls': 0 ,'params': b['params']})

            candidates = best_candidates
        
        sample_instances = random.sample(self.dataset.instances, 
                                            k=len(self.dataset.instances)//5)
            
        for candidate in candidates:
            candidate['val'] = 0
            candidate['oracle_calls'] = 0
        
        for instance in sample_instances:
            for candidate in candidates:
                self.try_parameters(candidate['params'])
                exp = self.explain(instance=instance)
                solution = self.minimize(exp)
              
                candidate['val'] += get_edge_differences(instance, solution)[0]
                candidate['oracle_calls'] += self.k
        
        self.logger.info("Final selecting")
        candidates.sort(key=lambda x: (x['val'], x['oracle_calls']))
        best_params = candidates[0]['params']
        self.try_parameters(best_params)

        self.logger.info("Parameters:")
        self.logger.info("neigh_factor:" + str(self.neigh_factor))
        self.logger.info("runtime_factor:" + str(self.runtime_factor))
        self.logger.info("max_runtime:" + str(self.max_runtime))
        self.logger.info("max_neigh:" + str(self.max_neigh))
        self.logger.info("max_oracle_calls:" + str(self.max_oracle_calls))

        self.logger.info("end train_parameters")