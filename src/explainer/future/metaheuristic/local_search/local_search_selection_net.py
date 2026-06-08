import copy
import math
import os
import random
from filelock import FileLock
import numpy as np
from src.core.explainer_base import Explainer
from src.dataset.instances.base import DataInstance
from src.dataset.instances.graph import GraphInstance
from src.explainer.future.meta.minimizer.base import ExplanationMinimizer
from src.explainer.future.metaheuristic.Tagging.OnlineSelector import OnlineNNEdgeSelector
from src.explainer.future.metaheuristic.Tagging.vectors_builder import VectorsBuilder
from typing import Generator

from src.explainer.future.metaheuristic.initial_solution_search.simple_searcher import SimpleSearcher
from src.explainer.future.metaheuristic.local_search.binary_model import BinaryModel
from src.explainer.future.metaheuristic.local_search.cache import FixedSizeCache
from src.explainer.future.metaheuristic.manipulation.methods import average_smoothing, average_smoothing_zero, feature_aggregation, heat_kernel_diffusion, identity, laplacian_regularization, random_walk_diffusion, weighted_smoothing
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
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

        self.searcher = SimpleSearcher()
        
        self.distance_metric = GraphEditDistanceMetric()  
        
        
        self.efficiency = 0.5
        # how fast efficiency penalizes high self.tries (bigger => less penalty)
        self.eff_tau = self.local_config['parameters'].get('eff_tau', 4)

        # smoothing factor for the moving average (0..1). bigger => reacts faster
        self.eff_alpha = self.local_config['parameters'].get('eff_alpha', 0.02)
        self.add_pool_size = 10000 
        self.total_tries = 0
        self.total_tries_steps = 0
        
        self.methods = [
            lambda data, features: identity(data, features),
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
        
        self.labels = []
        for i in range(self.G.num_nodes - 1):
            for j in range(i + 1, self.G.num_nodes):
                self.labels.append((i,j))
                
        self.label_to_id = {uv: idx for idx, uv in enumerate(self.labels)}
                
        metrics = [
            "degree",
            "local_clustering",
            "triangle_count",
            "coreness",
            "avg_neighbor_degree",
            "ego_density",
            "pagerank"
        ]

        vectorsBuilder = VectorsBuilder(metrics, self.G.data)
        metrics_features = vectorsBuilder.X
        
        node_features = instance.node_features
        total_features = np.concatenate((node_features, metrics_features), axis=1)
        
        # original_embeddings_tensors = self.oracle.get_node_embeddings(instance)
        
        # k = total_features.shape[1] + original_embeddings_tensors.shape[1]
        k = total_features.shape[1] 
        
        self.selector = self.load_or_initialize_selector(
                dataset_id=self.dataset.name,
                k=k,
                hidden_dim=64,
                num_hidden_layers=2,
                lr=1e-3,
                exploration_prob=0.1,
            )
                
        total_features_tensor = self.selector.build_tensors(total_features)
        
        # total_tensor = torch.cat((total_features_tensor, original_embeddings_tensors), dim=1)
        total_tensor = total_features_tensor

        
        self.selector.set_node_vectors_from_tensor(list(total_tensor))
        self.selector.set_base_graph(instance.data, directed=instance.directed)
        self.selector.add_pool_size = self.add_pool_size
        
        
        min_ctf = explaination.counterfactual_instances[0]

        
        _, diff_matrix = get_edge_differences(self.G, min_ctf)
        different_coordinates = np.where(diff_matrix == 1)        
        different_coords_list = list(zip(different_coordinates[0], different_coordinates[1]))
        # Filter to avoid duplicate edges in undirected graphs
        filtered_coords_list = [coord for coord in different_coords_list if coord[0] < coord[1]]
        actual = self.uv_to_id(filtered_coords_list)
        
        best = actual
        

        
        if(len(actual) == 0):
            return min_ctf
        
        self.cache = FixedSizeCache(capacity=500000)
        result = self.get_approximation(actual, best, min_ctf)
        
        return result
        
        
    def get_approximation(self, actual, best, min_ctf):
        self.logger.info("Initial solution: " + str(actual))
        self.logger.info("Initial solution size: " + str(len(actual)))

        result = min_ctf
        initial_solution = actual.copy()
        n = min(self.max_runtime, self.runtime_factor * len(actual))
        self.k = 0
        self.k_local = 0
        remove_move_examples = []
        add_move_examples = []
        p_neg_keep = 0.15  # keep 15% of failed moves
        flush_every = 30
        
        while(n > 0):
            # self.logger.info("n: " + str(n))
            self.logger.info("oracle calls (before (-)): " + str(self.k))
            n-=1
            self.k_local = 0
            if(len(best) == 1) : break
            if(self.k > self.max_oracle_calls or self.k_local > self.max_oracle_calls / 5) :
                 self.logger.info("Oracle calls limit reached, global: " + str(self.k) + ", local: " + str(self.k_local))
                 break
            found = False
            actual = best
            old_best_size = len(best)
            # self.logger.info("actual ---> " + str(len(actual)))
            
            self.tries = 0
            descarted = 0
            neg_removed_moves = []   # list of removed_uv (each is list[uv])
            neg_keep_first = 8       # always keep first failures (they’re “most confident” under greedy)
            for s, removed, _, sol_ctx in self.edge_remove(actual):
                if(self.k > self.max_oracle_calls or self.k_local > self.max_oracle_calls / 5) :
                 self.logger.info("Oracle calls limit reached, global: " + str(self.k) + ", local: " + str(self.k_local))
                 break
                if self.cache.contains(s):
                    descarted += 1
                    continue
                self.cache.add(s)
                # print("removed: " + str(removed))
                old_best_size = len(best)
                new_size = len(s)
                self.tries += 1
                
                found_, inst = self.evaluate(s)
                
                sol_uv = self.id_to_uv(sol_ctx)
                removed_uv = self.id_to_uv(removed)

                is_success = bool(found_ and new_size < old_best_size)
                
                # ---- Model training ----
                sol_uv = self.id_to_uv(sol_ctx)
                removed_uv = self.id_to_uv(removed)

                is_success = bool(found_ and new_size < old_best_size)

                if is_success:
                    # train: rank this removed_uv above the failures in THIS context
                    if neg_removed_moves:
                        self.selector.update_removal_ranked(sol_uv, removed_uv, neg_removed_moves)
                        neg_removed_moves = []
                else:
                    # keep early failures (hard negatives), and sample some later ones
                    if self.tries <= neg_keep_first or random.random() < p_neg_keep:
                        neg_removed_moves.append(removed_uv)
                # ------------------------

                if found_ and new_size < old_best_size:
                    neg_removed_moves = []
                    found = True
                    best = s
                    actual = s
                    result = inst
                        
                    n = min(self.max_runtime, self.runtime_factor * len(actual))
                    break

                
            if(found):
                self._update_efficiency_from_tries(tag="(-)")
                self.logger.info("============> (-) Found solution with size: " + str(len(actual)))
                continue
            
            half = int(len(actual) / 2)
            reduce = min(half, random.randint(1, half * 4))
            actual = self.reduce_random(best, reduce)
            # self.logger.info("actual ---> " + str(len(actual)))
            
            found = False
            
            while(len(best) - len(actual) > 1):
                if(self.k > self.max_oracle_calls or self.k_local > self.max_oracle_calls / 5) :
                    self.logger.info("Oracle calls limit reached, global: " + str(self.k) + ", local: " + str(self.k_local))
                    break
                n-=1
                self.logger.info("oracle calls (before (=)): " + str(self.k))
                self.tries = 0
                descarted = 0
                for s, removed, added, sol_ctx, temp_ctx in self.edge_swap(actual):
                    if(self.k > self.max_oracle_calls or self.k_local > self.max_oracle_calls / 5) :
                        self.logger.info("Oracle calls limit reached, global: " + str(self.k) + ", local: " + str(self.k_local))
                        break
                    if self.cache.contains(s):
                        descarted += 1
                        continue
                    self.cache.add(s)
                    self.tries += 1
                    old_best_size = len(best)
                    new_size = len(s)
                    
                    
                    found_, inst = self.evaluate(s)
                    
                    # ---- Model training ----
                    sol_uv = self.id_to_uv(sol_ctx)
                    temp_uv = self.id_to_uv(temp_ctx)
                    removed_uv = self.id_to_uv(removed)
                    added_uv = self.id_to_uv(added)

                    is_success = bool(found_ and new_size < old_best_size)

                    # removal move example
                    if removed_uv:
                        if is_success:
                            remove_move_examples.append((sol_uv, removed_uv, True))
                        else:
                            if random.random() < p_neg_keep:
                                remove_move_examples.append((sol_uv, removed_uv, False))

                    # addition move example
                    if added_uv:
                        if is_success:
                            add_move_examples.append((temp_uv, added_uv, True))
                        else:
                            if random.random() < p_neg_keep:
                                add_move_examples.append((temp_uv, added_uv, False))

                    if len(remove_move_examples) >= flush_every:
                        self.selector.update_removal_moves(remove_move_examples)
                        remove_move_examples = []

                    if len(add_move_examples) >= flush_every:
                        self.selector.update_addition_moves(add_move_examples)
                        add_move_examples = []
                    # ------------------------    
                    
                    if found_ and new_size < old_best_size:
                        found = True
                        best = s
                        actual = s
                        result = inst
                        
                        n = min(self.max_runtime, self.runtime_factor * len(actual))
                        break

                    
                    
                if(found):
                    self._update_efficiency_from_tries(tag="(=)")
                    self.logger.info("============> (=) Found solution with size: " + str(len(actual)))
                    break
                
                self.logger.info("oracle calls (before (+)): " + str(self.k))
                actual = self.reduce_random(best, len(actual))
                # self.logger.info("actual ===> " + str(len(actual)))
                
                self.tries = 0
                descarted = 0
                neg_added_moves = []
                for s, _, added, sol_ctx in self.edge_add(actual, best):
                    if(self.k > self.max_oracle_calls or self.k_local > self.max_oracle_calls / 5) :
                        self.logger.info("Oracle calls limit reached, global: " + str(self.k) + ", local: " + str(self.k_local))
                        break
                    if self.cache.contains(s):
                        descarted += 1
                        continue
                    self.cache.add(s)
                    self.tries += 1
                    old_best_size = len(best)
                    new_size = len(s)

                    found_, inst = self.evaluate(s)
                    
                    # ---- Model training ----
                    sol_uv = self.id_to_uv(sol_ctx)
                    added_uv = self.id_to_uv(added)

                    is_success = bool(found_ and new_size < old_best_size)

                    if is_success:
                        if neg_added_moves:
                            self.selector.update_addition_ranked(sol_uv, added_uv, neg_added_moves)
                            neg_added_moves = []
                    else:
                        if self.tries <= neg_keep_first or random.random() < p_neg_keep:
                            neg_added_moves.append(added_uv)

                    # ------------------------

                    if found_ and new_size < old_best_size:
                        found = True
                        best = s
                        actual = s
                        result = inst
                        n = min(self.max_runtime, self.runtime_factor * len(actual))

                        break

                    
                if(found):
                    self._update_efficiency_from_tries(tag="(+)")
                    self.logger.info("============> (+) Found solution with size: " + str(len(actual)))
                    break
                
                to_expand = int(((len(best) - len(actual)) / 2)) + 1
                expand = len(actual) + min(to_expand, random.randint(1, to_expand * 4))
                # self.logger.info("expand: " + str(expand) + ", best: " + str(len(best)))
                if(expand > len(best)): break
                actual = self.reduce_random(best, expand)
                # self.logger.info("actual +++> " + str(len(actual)))
            
            if found:
                continue
            break
        
        print("Oracle calls: " + str(self.k))
          
        if(self.oracle.predict(result) == self.oracle.predict(self.G)):
            self.logger.info("ERROR, returning non ctf ")
            self.logger.info("instance -> " + str(self.oracle.predict(self.G)))
            self.logger.info("result -> " + str(self.oracle.predict(result)))
        
        removed_edges = initial_solution - best
        added_edges = best - initial_solution
        self.logger.info("original: " + str(len(initial_solution)) + ", final: " + str(len(best)))
        self.logger.info("removed edges: " + str(len(removed_edges)) + ", added edges: " + str(len(added_edges)))
        
        init_uv = self.id_to_uv(initial_solution)
        best_uv = self.id_to_uv(best)

        removed_uv = self.id_to_uv(removed_edges)
        added_uv = self.id_to_uv(added_edges)

        if removed_uv:
            self.selector.update_removal_moves([(init_uv, removed_uv, True)])
        if added_uv:
            self.selector.update_addition_moves([(best_uv, added_uv, True)])
        
        if remove_move_examples:
            self.selector.update_removal_moves(remove_move_examples)
        if add_move_examples:
            self.selector.update_addition_moves(add_move_examples)
            
        self.save_selector(self.selector, self.dataset.name)
        return result
    
    def evaluate(self, solution : set[int]) -> tuple[bool, GraphInstance]:
        new_data = np.copy(self.G.data)
        self.disturb(new_data, self.G.directed, solution)
        
        # If the dataset has attributes in the nodes, then lets explore those with the methods
        if(self.attributed):
            for method in self.methods:
                self.k += 1
                self.k_local += 1
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
    
    
    ## ------ Neighborhood methods ----- ##
    def disturb(self, data, directed, solution : set[int]):
        for i in solution:
            (n1, n2) = self.labels[i]
            data[n1, n2] = (data[n1, n2] + 1) % 2
            if(not directed):
                data[n2, n1] = (data[n2, n1] + 1) % 2


    
    def reduce_random(self, solution : set[int], i: int) -> set[int]:
        if len(solution) < i:
            raise ValueError("The set does not have enough elements.")
        
        removed = self.selector.propose_removals(self.id_to_uv(solution), i)
        selected_elements = solution.difference(self.uv_to_id(removed))
        
        return selected_elements


    def edge_swap(self, solution : set[int]) -> Generator[set[int], set[int], set[int]]:
        cealing = min(len(solution), (self.EPlus - len(solution))) + 1
        step = int(cealing / self.max_neigh) + 1
        for i in range(1, cealing, step):
            self.tries = 0
            for _ in range(self.neigh_factor * 3):
                removed = self.selector.propose_removals(self.id_to_uv(solution), i)
                removed_set = self.uv_to_id(removed)
                temp_solution = solution.difference(removed_set)
                added = self.selector.propose_additions(self.id_to_uv(temp_solution), i)
                added_set = self.uv_to_id(added)
                new_s = temp_solution.union(added_set)
                
                yield [new_s, removed_set, added_set, solution, temp_solution]
                
    

    def edge_add(self, solution: set[int], best) -> Generator[set[int], set[int], set[int]]:
        uv_solution = self.id_to_uv(solution)

        ceiling = (len(best) - len(solution)) + 1
        step = int(ceiling / self.max_neigh) + 1

        explore_topk = 32  # tune: 16/32/64
        seed = random.randrange(1_000_000_000)

        # Build ranked pool ONCE for this snapshot
        add_order = self.selector.get_addition_trial_order(
            uv_solution,
            pool_size=None,         # uses selector.add_pool_size or default
            use_focus=True,
            explore_topk=explore_topk,
            seed=seed,
        )
        m = len(add_order)
        if m == 0:
            return

        for i in range(1, ceiling, step):
            self.tries = 0

            num_trials = self.neigh_factor * 3
            for t in range(num_trials):
                k = min(i, m)

                # produce multiple neighbors without rescoring:
                start = (t * k) % m
                end = start + k
                if end <= m:
                    added = add_order[start:end]
                else:
                    added = add_order[start:] + add_order[: end - m]

                added_set = self.uv_to_id(added)
                new_s = solution.union(added_set)

                yield [new_s, [], added_set, solution]


    def edge_remove(self, solution: set[int]) -> Generator[set[int], set[int], set[int]]:
        # Convert solution ids -> uv once
        uv_solution = self.id_to_uv(solution)
        ceiling = len(uv_solution)

        if ceiling == 0:
            return

        # Score ONCE + sort ONCE for this snapshot (cache inside selector handles repeats)
        # explore_topk: randomize within top-K once, but keep a mostly-fixed order.
        explore_topk = min(32, ceiling)  # tune (e.g. 16/32/64). Keep small for speed.
        seed = random.randrange(1_000_000_000)

        removal_order = self.selector.get_removal_trial_order(
            uv_solution,
            explore_topk=explore_topk,
            shuffle_topk_once=True,
            seed=seed,
            return_logits=False,
        )
        m = len(removal_order)
        if m == 0:
            return

        step = int((ceiling / self.max_neigh) + 1)

        for i in range(1, ceiling, step):
            self.tries = 0

            # Produce multiple neighbors WITHOUT rescoring:
            # take sliding windows / wrapped slices from the ranked order.
            num_trials = self.neigh_factor * 3
            for t in range(num_trials):
                k = min(i, m)
                if k == m:
                    removed = removal_order
                else:
                    # vary which edges are removed each trial, but preserve the ranked bias
                    start = (t * k) % m
                    end = start + k
                    if end <= m:
                        removed = removal_order[start:end]
                    else:
                        removed = removal_order[start:] + removal_order[: end - m]

                removed_set = self.uv_to_id(removed)
                new_s = solution.difference(removed_set)

                yield [new_s, removed_set, [], solution]



    ## ------ Selector persistence helpers ----- ##
    def id_to_uv(self, ids: set[int]) -> list[(int, int)]:
        result = []
        for i in ids:
            result.append(self.labels[i])
        return result
    
    def uv_to_id(self, uv: list[tuple[int, int]]) -> set[int]:
        return { self.label_to_id[(i, j) if i < j else (j, i)] for (i, j) in uv if i != j }
    
    

    
    def model_paths(self, dataset_id: str):
        path = f"models/edge_selector_{dataset_id}.pt"
        lock_path = path + ".lock"
        return path, lock_path


    def load_or_initialize_selector(self, dataset_id: str, k: int, **selector_kwargs):
        """
        Safely load (or create) a selector for a dataset.
        Caller still needs to call set_node_vectors() afterwards.

        selector_kwargs can include:
            - hidden_dim (int)
            - num_hidden_layers (int)
            - lr (float)
            - exploration_prob (float)
            - device (torch.device)
        """
        hidden_dim = selector_kwargs.get("hidden_dim", 128)
        num_hidden_layers = selector_kwargs.get("num_hidden_layers", 2)
        lr = selector_kwargs.get("lr", 1e-3)
        exploration_prob = selector_kwargs.get("exploration_prob", 0.2)
        device = selector_kwargs.get("device", None)

        path, lock_path = self.model_paths(dataset_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with FileLock(lock_path):
            if os.path.exists(path):
                # Load existing model (architecture comes from checkpoint)
                selector = OnlineNNEdgeSelector.load(path, lr=lr, device=device)

                selector.exploration_prob = exploration_prob

            else:
                # Create a new model with the given architecture
                selector = OnlineNNEdgeSelector(
                    k=k,
                    hidden_dim=hidden_dim,
                    num_hidden_layers=num_hidden_layers,
                    lr=lr,
                    exploration_prob=exploration_prob,
                    device=device,
                )
                selector.save(path)  # create initial checkpoint

        return selector
    

    def save_selector(self, selector: OnlineNNEdgeSelector, dataset_id: str):
        """
        Safely save a selector for a dataset.
        """
        path, lock_path = self.model_paths(dataset_id)
        with FileLock(lock_path):
            selector.save(path) 
            
           
    def _update_efficiency_from_tries(self, tag: str):
        self.total_tries += self.tries
        self.total_tries_steps += 1
        # normalized score: self.tries=0 -> 1.0, self.tries grows -> approaches 0.0
        tau = max(1e-9, float(self.eff_tau))
        score = 1.0 / (1.0 + (max(0, int(self.tries)) / tau))   # in (0,1]

        old = float(self.efficiency)
        a = float(self.eff_alpha)
        self.efficiency = max(0.0, min(1.0, (1.0 - a) * old + a * score))

        self.logger.info(
            f"{tag}: tries={self.tries}, Efficiency={self.efficiency:.4f}, AvgTries={self.total_tries / self.total_tries_steps:.2f}"
        )
    
    def write(self):
        pass

    def read(self):
        pass