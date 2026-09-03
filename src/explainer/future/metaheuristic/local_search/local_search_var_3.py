"""LBS ordering ablation - variant 3: order  add -> swap -> 1a -> 1b.

E3 of REVISION_EXPERIMENTS.md (R2.11) permutes the strategy priority while
keeping every operator. Per outer iteration the operators are tried in the
order: edge_add, edge_swap, single-edge removal (1a), then overshoot removal
(1b, remove a large block at once). Removal is deprioritised and, when it
runs, it escalates fine -> coarse (single edge first, big block last). The
first improving candidate is accepted and the search restarts from it.

Building blocks (shared with var_4, which only changes the order):
  1a = edge_remove_fine     (remove a single edge)
  1b = edge_remove_overshoot(remove a large block at once)
  add/swap operate on a random reduction of best.
"""
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
from src.explainer.future.metaheuristic.manipulation.methods import average_smoothing, average_smoothing_zero, feature_aggregation, heat_kernel_diffusion, identity, laplacian_regularization, random_walk_diffusion, weighted_smoothing
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.utils.cfg_utils import init_dflts_to_of
from src.utils.comparison import get_edge_differences
from src.utils.metrics.ged import GraphEditDistanceMetric
from src.utils.seeding import set_seed
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
        # Opt-in (hash-stable): skip per-candidate dataset.manipulate() when
        # the oracle ignores recomputed node features (e.g. ASD, Tree-Cycles).
        self.recompute_features = self.local_config['parameters'].get('recompute_features', True)

        set_seed(self.local_config['parameters'].get('seed'))

        self.tagger = SimpleTagger()
        self.searcher = SimpleSearcher()
        self.distance_metric = GraphEditDistanceMetric()

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
        self.EPlus = int((self.N * (self.N - 1)) / 2)

        self.M = BinaryModel(self.oracle, instance)
        self.labels = self.tagger.tag(instance)

        min_ctf = explaination.counterfactual_instances[0]

        _, diff_matrix = get_edge_differences(self.G, min_ctf)
        different_coordinates = np.where(diff_matrix == 1)
        different_coords_list = list(zip(different_coordinates[0], different_coordinates[1]))
        filtered_coords_list = [c for c in different_coords_list if c[0] < c[1]]
        actual = self.tagger.get_indices(self.labels, filtered_coords_list)
        best = actual

        if len(actual) == 0:
            self.logger.info("Initial solution size is 0")
            return min_ctf

        if self.oracle.predict(min_ctf) == self.oracle.predict(self.G):
            self.logger.info("Generator was incapable of finding counterfactual, returning non ctf")
            return min_ctf

        self.cache = FixedSizeCache(capacity=500000)
        return self.get_approximation(actual, best, min_ctf)

    def get_approximation(self, actual, best, min_ctf):
        self.logger.info("Initial solution: " + str(actual))
        self.logger.info("Initial solution size: " + str(len(actual)))

        result = min_ctf
        n = min(self.max_runtime, self.runtime_factor * len(actual))
        self.k = 0
        while n > 0:
            n -= 1
            if len(best) == 1:
                break
            if self.k > self.max_oracle_calls:
                self.logger.info("Oracle calls limit reached")
                break

            # Reduced starting point for the repair operators (add / swap).
            half = int(len(best) / 2)
            if half >= 1:
                keep = min(half, random.randint(1, half * 4))
                reduced = self.reduce_random(best, keep)
            else:
                reduced = set(best)

            # Ordering ablation: phases are tried in this variant's order;
            # the first improving candidate is accepted and the search
            # restarts from it. If no phase improves, the loop retries with a
            # fresh stochastic reduction (n decrements guarantee termination).
            phases = [
                ("(+)", self.edge_add(reduced, best)),
                ("(=)", self.edge_swap(reduced)),
                ("(1a)", self.edge_remove_fine(best)),
                ("(1b)", self.edge_remove_overshoot(best)),
            ]
            for tag, gen in phases:
                ok, s, inst = self._scan(gen, best)
                if ok:
                    best, result = s, inst
                    n = min(self.max_runtime, self.runtime_factor * len(best))
                    self.logger.info("============> " + tag + " size: " + str(len(best)))
                    break

        if self.oracle.predict(result) == self.oracle.predict(self.G):
            self.logger.info("ERROR, returning non ctf ")
            self.logger.info("instance -> " + str(self.oracle.predict(self.G)))
            self.logger.info("result -> " + str(self.oracle.predict(result)))
        return result


    def evaluate(self, solution):
        new_data = np.copy(self.G.data)
        self.disturb(new_data, self.G.directed, solution)

        if self.attributed:
            for method in self.methods:
                self.k += 1
                node_features = method(new_data, self.G.node_features)
                new_g = GraphInstance(id=self.G.id, label=0, data=new_data,
                                      directed=self.G.directed,
                                      node_features=node_features)
                if self.M.classify(new_g):
                    return (True, new_g)
        else:
            self.k += 1
            new_g = GraphInstance(id=self.G.id, label=0, data=new_data,
                                  directed=self.G.directed,
                                  node_features=self.G.node_features)
            if self.recompute_features:
                self.dataset.manipulate(new_g)
            if self.M.classify(new_g):
                return (True, new_g)

        return (False, None)

    def disturb(self, data, directed, solution):
        for i in solution:
            (n1, n2) = self.labels[i]
            data[n1, n2] = (data[n1, n2] + 1) % 2
            if not directed:
                data[n2, n1] = (data[n2, n1] + 1) % 2

    def swap_random(self, solution, i):
        self.remove_random(solution, i)
        self.add_random(solution, i)
        return solution

    def add_random(self, solution, i):
        available = set(range(1, self.EPlus)) - solution
        if len(available) < i:
            raise ValueError("Not enough available numbers to add.")
        solution.update(random.sample(list(available), i))
        return solution

    def remove_random(self, solution, i):
        solution.difference_update(random.sample(list(solution), i))
        return solution

    def reduce_random(self, solution, i):
        if len(solution) < i:
            raise ValueError("The set does not have enough elements.")
        return set(random.sample(list(solution), i))

    def edge_swap(self, solution):
        cealing = min(len(solution), (self.EPlus - len(solution))) + 1
        step = int(cealing / self.max_neigh) + 1
        for i in range(1, cealing, step):
            for _ in range(self.neigh_factor ** 2):
                yield self.swap_random(set(solution), i)

    def edge_add(self, solution, best):
        cealing = (len(best) - len(solution)) + 1
        step = int(cealing / self.max_neigh) + 1
        for i in range(1, cealing, step):
            for _ in range(self.neigh_factor ** 2):
                yield self.add_random(set(solution), i)

    def edge_remove(self, solution):
        cealing = len(solution)
        step = int((cealing / self.max_neigh) + 1)
        for i in range(0, cealing, step):
            for _ in range(self.neigh_factor ** 3):
                yield self.remove_random(set(solution), i)

    def _scan(self, candidates, best):
        """First candidate that is a valid CF strictly smaller than best."""
        for s in candidates:
            if self.cache.contains(s):
                continue
            self.cache.add(s)
            found_, inst = self.evaluate(s)
            if found_ and len(s) < len(best):
                return True, s, inst
        return False, None, None

    def edge_remove_fine(self, solution):
        # 1a: remove a single edge at a time (conservative removal).
        for _ in range(self.neigh_factor ** 3):
            yield self.remove_random(set(solution), 1)

    def edge_remove_overshoot(self, solution):
        # 1b: remove a large block of edges at once (overshoot), largest first.
        cealing = len(solution)
        if cealing <= 2:
            return
        step = int((cealing / self.max_neigh) + 1)
        for i in range(cealing - 1, 1, -step):
            for _ in range(self.neigh_factor ** 3):
                yield self.remove_random(set(solution), i)

    def write(self):
        pass

    def read(self):
        pass
