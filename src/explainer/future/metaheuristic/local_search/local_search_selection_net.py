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
from src.explainer.future.metaheuristic.Tagging.ann import ANNIndexWeighted
from src.explainer.future.metaheuristic.Tagging.vectors_builder import VectorsBuilder
from typing import Generator

from src.explainer.future.metaheuristic.initial_solution_search.simple_searcher import SimpleSearcher
from src.explainer.future.metaheuristic.local_search.binary_model import BinaryModel
from src.explainer.future.metaheuristic.local_search.cache import FixedSizeCache
from src.explainer.future.metaheuristic.manipulation.methods import average_smoothing, feature_aggregation, heat_kernel_diffusion, laplacian_regularization, random_walk_diffusion, weighted_smoothing
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
import torch
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
        
        # --- stats for logging ---
        self.stats_total_moves = 0              # how many neighbor moves evaluated
        self.stats_accepted_moves = 0           # how many were accepted (improved best)
        self.stats_total_improvement = 0.0      # sum of (old_size - new_size) for accepted moves
        self.stats_total_confidence = 0.0       # sum of model_confidence over all moves

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
        
        
        
        metrics = [
            "degree",
            "closeness",
            "eigenvector",
            "betweenness",
            "pagerank",
            "component_id",
            "eccentricity",
            "local_clustering",     
            "triangle_count",
        ]

        vectorsBuilder = VectorsBuilder(metrics, self.G.data)
        node_features = np.concatenate((instance.node_features, vectorsBuilder.X), axis=1)
        
        print("Vectors builder shape: " + str(vectorsBuilder.X.shape))
        print("Original node features shape: " + str(instance.node_features.shape))
        print("New node features shape: " + str(node_features.shape))
        
        K = node_features.shape[1]
        print("Node feature dimension: " + str(K))
        self.selector = self.load_or_initialize_selector(
                dataset_id=self.dataset.name,
                k=K,
                hidden_dim=64,
                num_hidden_layers=2,
                lr=2e-3,
                exploration_prob=0.3,
            )

        self.selector.set_node_vectors(node_features)  # convert once to torch
        
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
        self.log_selector_metrics(prefix="[INIT] ")

        n = min(self.max_runtime, self.runtime_factor * len(actual))
        self.k = 0
        while(n > 0):
            self.selector.exploration_prob = max(0.05, self.selector.exploration_prob * 0.995)
            self.log_selector_metrics(prefix="[progress] ")
            # self.logger.info("n: " + str(n))
            # self.logger.info("k: " + str(self.k))
            n-=1
            if(len(best) == 1) : break
            if(self.k > self.max_oracle_calls) :
                 self.logger.info("Oracle calls limit reached")
                 break
            found = False
            actual = best
            old_best_size = len(best)
            # self.logger.info("actual ---> " + str(len(actual)))
            
            for s, removed, _ in self.edge_remove(actual):
                if self.cache.contains(s):
                    continue
                self.cache.add(s)
                # print("removed: " + str(removed))
                old_best_size = len(best)
                new_size = len(s)

                found_, inst = self.evaluate(s)
                reward = self.compute_reward(old_best_size, new_size, found_)

                if found_ and new_size < old_best_size:
                    found = True
                    best = s
                    actual = s
                    result = inst
                    
                    # reward > 0 here
                    if(min(self.max_runtime, self.runtime_factor * len(actual)) -1 == n):
                        self._update_search_stats(
                            old_best_size=old_best_size,
                            new_size=len(s),
                            success=True,
                            chosen_pairs_uv=self.id_to_uv(removed),
                            mode="remove",
                        )
                        
                    n = min(self.max_runtime, self.runtime_factor * len(actual))
                    self.selector.update_removals(self.id_to_uv(removed), reward)
                    break
                else:
                    # reward will be 0 for non-improvements / invalid
                    self.selector.update_removals(self.id_to_uv(removed), reward)
                    if(min(self.max_runtime, self.runtime_factor * len(actual)) -1 == n):
                        self._update_search_stats(
                            old_best_size=old_best_size,
                            new_size=len(s),
                            success=False,
                            chosen_pairs_uv=self.id_to_uv(removed),
                            mode="remove",
                        )
                
            if(found):
                self.logger.info("============> (-) Found solution with size: " + str(len(actual)))
                continue
            
            half = int(len(actual) / 2)
            reduce = min(half, random.randint(1, half * 4))
            actual = self.reduce_random(best, reduce)
            # self.logger.info("actual ---> " + str(len(actual)))
            
            while(len(best) - len(actual) > 1):
                n-=1
                for s, removed, added in self.edge_swap(actual):
                    if self.cache.contains(s):
                        continue
                    self.cache.add(s)

                    old_best_size = len(best)
                    new_size = len(s)

                    found_, inst = self.evaluate(s)
                    reward = self.compute_reward(old_best_size, new_size, found_)

                    if found_ and new_size < old_best_size:
                        found = True
                        best = s
                        actual = s
                        result = inst
                        
                        if(min(self.max_runtime, self.runtime_factor * len(actual)) -1 == n):
                            self._update_search_stats(
                                    old_best_size=old_best_size,
                                    new_size=len(s),
                                    success=True,
                                    chosen_pairs_uv=self.id_to_uv(removed),
                                    mode="remove",
                                )
                            self._update_search_stats(
                                    old_best_size=old_best_size,
                                    new_size=len(s),
                                    success=True,
                                    chosen_pairs_uv=self.id_to_uv(added),
                                    mode="add",
                                )
                        
                        n = min(self.max_runtime, self.runtime_factor * len(actual))
                        self.selector.update_additions(self.id_to_uv(added), reward)
                        self.selector.update_removals(self.id_to_uv(removed), reward)
                        break
                    else:
                        if(min(self.max_runtime, self.runtime_factor * len(actual)) -1 == n):
                            self._update_search_stats(
                                    old_best_size=old_best_size,
                                    new_size=len(s),
                                    success=False,
                                    chosen_pairs_uv=self.id_to_uv(removed),
                                    mode="remove",
                                )
                            self._update_search_stats(
                                    old_best_size=old_best_size,
                                    new_size=len(s),
                                    success=False,
                                    chosen_pairs_uv=self.id_to_uv(added),
                                    mode="add",
                                )
                        self.selector.update_additions(self.id_to_uv(added), reward)
                        self.selector.update_removals(self.id_to_uv(removed), reward)

                    
                    
                if(found):
                    self.logger.info("============> (=) Found solution with size: " + str(len(actual)))
                    break

                actual = self.reduce_random(best, len(actual))
                # self.logger.info("actual ===> " + str(len(actual)))
                
                for s, _, added in self.edge_add(actual, best):
                    if self.cache.contains(s):
                        continue
                    self.cache.add(s)

                    old_best_size = len(best)
                    new_size = len(s)

                    found_, inst = self.evaluate(s)
                    reward = self.compute_reward(old_best_size, new_size, found_)

                    if found_ and new_size < old_best_size:
                        found = True
                        best = s
                        actual = s
                        result = inst
                        if(min(self.max_runtime, self.runtime_factor * len(actual)) -1 == n):
                            self._update_search_stats(
                                old_best_size=old_best_size,
                                new_size=len(s),
                                success=True,
                                chosen_pairs_uv=self.id_to_uv(added),
                                mode="add",
                            )
                        n = min(self.max_runtime, self.runtime_factor * len(actual))
                        
                        self.selector.update_additions(self.id_to_uv(added), reward)
                        break
                    else:
                        if(min(self.max_runtime, self.runtime_factor * len(actual)) -1 == n):
                            self._update_search_stats(
                                    old_best_size=old_best_size,
                                    new_size=len(s),
                                    success=False,
                                    chosen_pairs_uv=self.id_to_uv(added),
                                    mode="add",
                                )
                        self.selector.update_additions(self.id_to_uv(added), reward)

                    
                if(found):
                    self.logger.info("============> (+) Found solution with size: " + str(len(actual)))
                    break
                
                to_expand = int(((len(best) - len(actual)) / 2)) + 1
                expand = len(actual) + min(to_expand, random.randint(1, to_expand * 4))
                # self.logger.info("expand: " + str(expand) + ", best: " + str(len(best)))
                if(expand > len(best)): break
                actual = self.reduce_random(best, expand)
                # self.logger.info("actual +++> " + str(len(actual)))
          
        if(self.oracle.predict(result) == self.oracle.predict(self.G)):
            self.logger.info("ERROR, returning non ctf ")
            self.logger.info("instance -> " + str(self.oracle.predict(self.G)))
            self.logger.info("result -> " + str(self.oracle.predict(result)))
        
        self.save_selector(self.selector, self.dataset.name)
        self.log_selector_metrics(prefix="[FINAL] ")
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
        
        selected_elements = set(random.sample(list(solution), i))
        
        return selected_elements


    def edge_swap(self, solution : set[int]) -> Generator[set[int], set[int], set[int]]:
        cealing = min(len(solution), (self.EPlus - len(solution))) + 1
        step = int(cealing / self.max_neigh) + 1
        for i in range(1, cealing, step):
            for _ in range(self.neigh_factor ** 2):
                removed = self.selector.propose_removals(self.id_to_uv(solution), i)
                removed_set = self.uv_to_id(removed)
                temp_solution = solution.difference(removed_set)
                added = self.selector.propose_additions(self.id_to_uv(temp_solution), i)
                added_set = self.uv_to_id(added)
                new_s = temp_solution.union(added_set)
                yield [new_s, removed_set, added_set]
                
    

    def edge_add(self, solution : set[int], best) -> Generator[set[int], set[int], set[int]]:
        cealing = (len(best) - len(solution)) + 1
        step = int(cealing / self.max_neigh) + 1
        for i in range(1, cealing, step):
            for _ in range(self.neigh_factor ** 2):
                added = self.selector.propose_additions(self.id_to_uv(solution), i)
                added_set = self.uv_to_id(added)
                new_s = solution.union(added_set)
                yield [new_s, [], added_set]

    def edge_remove(self, solution : set[int]) -> Generator[set[int], set[int], set[int]]:
        cealing = len(solution)
        step = int((cealing / self.max_neigh) + 1) 
        for i in range(1, cealing, step):
            for _ in range(self.neigh_factor ** 3):
                removed = self.selector.propose_removals(self.id_to_uv(solution), i)
                removed_set = self.uv_to_id(removed)
                new_s = solution.difference(removed_set)
                yield [new_s, removed_set, []]


    ## ------ Selector persistence helpers ----- ##
    def id_to_uv(self, ids: set[int]) -> list[(int, int)]:
        result = []
        for i in ids:
            result.append(self.labels[i])
        return result
    
    def uv_to_id(self, uv: list[(int, int)]) -> set[int]:
        result = set()
        for (i, j) in uv:
            result.add(self.labels.index((i, j)))
        return result
    
    

    
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
            
            
    ## ------ Logging helpers ----- ##
    
    def log_selector_metrics(self, prefix: str = ""):
        """
        Log accept_rate, avg_improvement, and model_confidence
        accumulated so far in this LocalSearch run.
        """
        total_moves = getattr(self, "stats_total_moves", 0)
        accepted_moves = getattr(self, "stats_accepted_moves", 0)
        total_improvement = getattr(self, "stats_total_improvement", 0.0)
        total_confidence = getattr(self, "stats_total_confidence", 0.0)

        if total_moves == 0:
            accept_rate = 0.0
            avg_improvement = 0.0
            model_conf = 0.0
        else:
            accept_rate = accepted_moves / total_moves
            avg_improvement = (
                total_improvement / accepted_moves if accepted_moves > 0 else 0.0
            )
            model_conf = total_confidence / total_moves

        msg = (
            f"{prefix}selector_stats | "
            f"moves={total_moves} "
            f"accept_rate={accept_rate:.4f} "
            f"avg_improvement={avg_improvement:.4f} "
            f"model_confidence={model_conf:.4f} "
            f"exploration_prob={self.selector.exploration_prob:.4f}"
        )
        self.logger.info(msg)


    def _selector_confidence(self, uv_pairs: list[tuple[int, int]], mode: str) -> float:
        """
        Average predicted probability of success for the given pairs, according
        to the current selector and mode ('add' or 'remove').
        """
        if not uv_pairs:
            return 0.0

        # Build a tensor of indices on the same device as the selector
        pairs_tensor = torch.tensor(
            uv_pairs,
            dtype=torch.long,
            device=self.selector.device,
        )

        model = self.selector.model_add if mode == "add" else self.selector.model_remove
        model.eval()
        with torch.no_grad():
            feats = self.selector._pair_features(pairs_tensor)
            logits = model(feats)
            probs = torch.sigmoid(logits)  # (batch,)
            return float(probs.mean().item())
        
    def _update_search_stats(
        self,
        old_best_size: int,
        new_size: int,
        success: bool,
        chosen_pairs_uv: list[tuple[int, int]],
        mode: str,
    ):
        """
        Update cumulative stats for logging.

        old_best_size: size of best solution before this move
        new_size: size of candidate solution
        success: True if this move produced a new best solution
        chosen_pairs_uv: list of (u, v) pairs that were added/removed in this move
        mode: 'add' or 'remove' (for model_confidence)
        """
        self.stats_total_moves += 1

        if success:
            self.stats_accepted_moves += 1
            improvement = max(0, old_best_size - new_size)
            self.stats_total_improvement += improvement

        # model_confidence: average predicted p(success) for chosen pairs
        conf = self._selector_confidence(chosen_pairs_uv, mode)
        self.stats_total_confidence += conf
        
    def compute_reward(self, old_size: int, new_size: int, found_: bool) -> float:
        """
        Map (valid move, size improvement) -> reward in [0, 1].

        - If the new solution is not a valid counterfactual (found_ == False) -> reward 0.
        - If new_size >= old_size -> no improvement -> reward 0.
        - Otherwise: reward proportional to relative size reduction.
        """
        if not found_:
            return 0.0
        if new_size >= old_size:
            return 0.0

        delta = old_size - new_size  # positive
        # normalize by old_size to get something in (0,1]
        reward = delta / old_size
        # clamp safely
        return float(max(0.0, min(1.0, reward)))

    def write(self):
        pass

    def read(self):
        pass