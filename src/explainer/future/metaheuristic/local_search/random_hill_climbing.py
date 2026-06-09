import random
import numpy as np

from src.dataset.instances.base import DataInstance
from src.dataset.instances.graph import GraphInstance
from src.explainer.future.meta.minimizer.base import ExplanationMinimizer
from src.explainer.future.metaheuristic.Tagging.simple_tagger import SimpleTagger
from src.explainer.future.metaheuristic.initial_solution_search.simple_searcher import SimpleSearcher
from src.explainer.future.metaheuristic.local_search.binary_model import BinaryModel
from src.explainer.future.metaheuristic.local_search.cache import FixedSizeCache
from src.explainer.future.metaheuristic.manipulation.methods import (
    average_smoothing,
    feature_aggregation,
    heat_kernel_diffusion,
    identity,
    laplacian_regularization,
    random_walk_diffusion,
    weighted_smoothing,
)
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.utils.comparison import get_edge_differences
from src.utils.metrics.ged import GraphEditDistanceMetric
from src.utils.seeding import set_seed


class RandomHillClimbing(ExplanationMinimizer):
    """Random-restart hill-climbing baseline.

    Drop-in sibling of :class:`LocalSearch`: same constructor surface, same
    minimizer interface, same oracle-call budget knob (``max_oracle_calls``).
    The contrast with LBS is the search strategy: move type is sampled
    uniformly from {delete, swap, add}, acceptance is greedy, and when
    ``patience`` consecutive steps pass with no improvement the search
    restarts from a random reduction of the best solution found so far.

    Same budget head-to-head against LBS isolates the value of the priority
    strategy (R1.3).
    """

    def check_configuration(self):
        super().check_configuration()
        params = self.local_config['parameters']
        if 'attributed' not in params:
            params['attributed'] = False
        if 'manip_attr' not in params:
            params['manip_attr'] = False
        if 'max_oracle_calls' not in params:
            params['max_oracle_calls'] = 10000
        if 'patience' not in params:
            params['patience'] = 40

    def init(self):
        super().init()
        self.logger = self.context.logger
        params = self.local_config['parameters']
        self.attributed = params['attributed']
        self.manip_attr = params['manip_attr']
        self.max_oracle_calls = params['max_oracle_calls']
        self.patience = params['patience']

        # Opt-in deterministic seeding (Note C). Legacy configs that omit
        # ``seed`` keep their hash and stay non-deterministic as before.
        set_seed(params.get('seed'))

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
            lambda data, features: random_walk_diffusion(data, features, steps=1),
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
        self.k = 0
        steps_since_improve = 0
        move_types = ("del", "swap", "add")

        while self.k < self.max_oracle_calls:
            if len(best) <= 1:
                break

            if steps_since_improve >= self.patience:
                target = random.randint(1, len(best) - 1)
                actual = self.reduce_random(set(best), target)
                steps_since_improve = 0
                self.logger.info("============> (restart) size: " + str(len(actual)))

            move = random.choice(move_types)
            candidate = None

            if move == "del" and len(actual) > 0:
                i = random.randint(1, len(actual))
                candidate = self.remove_random(set(actual), i)
            elif move == "swap" and 1 <= len(actual) < self.EPlus:
                i_max = min(len(actual), self.EPlus - len(actual))
                if i_max >= 1:
                    i = random.randint(1, i_max)
                    candidate = self.swap_random(set(actual), i)
            elif move == "add" and len(actual) < len(best):
                gap = len(best) - len(actual)
                i = random.randint(1, gap)
                candidate = self.add_random(set(actual), i)

            if candidate is None:
                steps_since_improve += 1
                continue

            if self.cache.contains(candidate):
                continue
            self.cache.add(candidate)

            found_, inst = self.evaluate(candidate)
            if found_ and len(candidate) < len(best):
                best = candidate
                actual = candidate
                result = inst
                steps_since_improve = 0
                self.logger.info("============> (rand-" + move + ") size: " + str(len(actual)))
            else:
                steps_since_improve += 1

        if self.oracle.predict(result) == self.oracle.predict(self.G):
            self.logger.info("ERROR, returning non ctf ")
        return result

    def evaluate(self, solution):
        new_data = np.copy(self.G.data)
        self.disturb(new_data, self.G.directed, solution)

        if self.attributed and self.manip_attr:
            for method in self.methods:
                self.k += 1
                node_features = method(new_data, self.G.node_features)
                new_g = GraphInstance(id=self.G.id,
                                      label=0,
                                      data=new_data,
                                      directed=self.G.directed,
                                      node_features=node_features)
                if self.M.classify(new_g):
                    return (True, new_g)
        else:
            self.k += 1
            new_g = GraphInstance(id=self.G.id,
                                  label=0,
                                  data=new_data,
                                  directed=self.G.directed,
                                  node_features=self.G.node_features)
            if not self.attributed:
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

    def add_random(self, solution, i):
        available = set(range(1, self.EPlus)) - solution
        if len(available) < i:
            raise ValueError("Not enough available numbers to add.")
        solution.update(random.sample(list(available), i))
        return solution

    def remove_random(self, solution, i):
        solution.difference_update(random.sample(list(solution), i))
        return solution

    def swap_random(self, solution, i):
        self.remove_random(solution, i)
        self.add_random(solution, i)
        return solution

    def reduce_random(self, solution, i):
        if len(solution) < i:
            raise ValueError("The set does not have enough elements.")
        return set(random.sample(list(solution), i))

    def write(self):
        pass

    def read(self):
        pass
