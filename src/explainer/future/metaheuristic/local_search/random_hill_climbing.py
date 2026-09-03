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
    """Vanilla random-restart hill-climbing baseline (Exp. 3b / R1.3).

    By design this is the simplest possible baseline against LBS, so it
    deliberately avoids every LBS-specific search trick:

    * **No heuristic neighborhood structure** (no ``del`` / ``swap`` /
      ``add`` strategy dispatch, no priority ordering between them).
      A single random bit in the edge-edit set is flipped per step.
    * **No bounded-step heuristics** (no ``gap = best - actual``
      constraint, no ``runtime_factor * len(actual)`` shrinking
      schedule, no neighborhood ceiling). The bit to flip is sampled
      uniformly from the full edge space.
    * **No multi-edge batched moves**. One bit per iteration.

    What it keeps from the canonical RRHC algorithm: greedy acceptance
    (strict improvement only), restarts after ``patience`` evaluated
    candidates without improvement, and an oracle-call budget for fair
    comparison against LBS.

    Restarts are drawn from random subsets of the incumbent ``best``
    (NOT from a uniform random state: with 2^EPlus states a uniform
    restart would be a strawman that never lands anywhere useful). This
    down-set restart is the only structural information the baseline
    uses, and it cuts in the baseline's favor.

    Implementation efficiencies that keep the comparison fair without
    adding domain heuristics: candidates whose size already fails the
    acceptance test (``len >= len(best)``) are rejected by free
    arithmetic BEFORE paying an oracle call, and already-evaluated
    candidates are deduplicated by a cache (same cache LBS uses). The
    oracle budget is the binding stop; a generous total-iteration cap
    only guards against spinning on a fully cached neighborhood at zero
    oracle cost.
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
        # Opt-in (hash-stable): skip per-candidate dataset.manipulate() when
        # the oracle ignores recomputed node features (e.g. ASD, Tree-Cycles).
        self.recompute_features = params.get('recompute_features', True)
        self.patience = params['patience']
        # Hash-stable opt-in. Default lets the oracle budget be the binding
        # stop (canonical RRHC termination): with patience evaluated
        # candidates per window, the budget can fund at most
        # max_oracle_calls // patience windows, so the give-up never fires
        # before the budget unless explicitly lowered in the config.
        self.max_restarts_no_improve = params.get(
            'max_restarts_no_improve',
            max(1, self.max_oracle_calls // max(1, self.patience)))

        # Opt-in deterministic seeding (Note C). Legacy configs that omit
        # ``seed`` keep their hash and stay non-deterministic as before.
        set_seed(params.get('seed'))

        self.tagger = SimpleTagger()
        self.searcher = SimpleSearcher()
        self.distance_metric = GraphEditDistanceMetric()

        # ``self.methods`` is only consulted when ``attributed`` and
        # ``manip_attr`` are both True, i.e. when the user explicitly
        # opts into the LBS-style attribute manipulation. The vanilla
        # baseline runs with ``manip_attr=False``.
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
        actual = set(self.tagger.get_indices(self.labels, filtered_coords_list))
        best = set(actual)

        if len(best) == 0:
            self.logger.info("Initial solution size is 0")
            return min_ctf

        if self.oracle.predict(min_ctf) == self.oracle.predict(self.G):
            self.logger.info("Generator was incapable of finding counterfactual, returning non ctf")
            return min_ctf

        self.cache = FixedSizeCache(capacity=500000)
        return self.get_approximation(actual, best, min_ctf)

    def get_approximation(self, actual, best, min_ctf):
        self.logger.info("Initial solution size: " + str(len(best)))

        result = min_ctf
        self.k = 0
        steps_since_improve = 0
        restarts_without_improve = 0
        # Free-spin guard: bounds loop iterations that cost no oracle calls
        # (cache hits, size-rejections). Generous on purpose; the oracle
        # budget is the intended binding stop.
        iterations = 0
        max_iterations = 50 * self.max_oracle_calls

        while self.k < self.max_oracle_calls and iterations < max_iterations:
            iterations += 1
            if len(best) <= 1:
                break

            if steps_since_improve >= self.patience:
                restarts_without_improve += 1
                if restarts_without_improve > self.max_restarts_no_improve:
                    self.logger.info(
                        "============> (giving up after "
                        f"{self.max_restarts_no_improve} restarts without improvement)"
                    )
                    break
                target = random.randint(1, max(1, len(best) - 1))
                actual = self._random_subset(best, target)
                steps_since_improve = 0
                self.logger.info("============> (restart) size: " + str(len(actual)))

            # Single random bit flip in the full edge-edit space. No move
            # type, no step-size heuristic, no bounding by ``best``. Range
            # is ``[0, EPlus)`` to match :class:`SimpleTagger`'s 0-indexed
            # labels (LBS's own helpers skip label 0; vanilla doesn't).
            bit = random.randrange(0, self.EPlus)
            candidate = set(actual)
            if bit in candidate:
                candidate.remove(bit)
            else:
                candidate.add(bit)

            if not candidate:
                continue
            # Free part of the objective first: a candidate at least as large
            # as the incumbent can never satisfy the acceptance test, so do
            # not pay an oracle call for it (arithmetic short-circuit, not a
            # domain heuristic). Not cached (never evaluated) and not counted
            # toward patience (no information gained).
            if len(candidate) >= len(best):
                continue
            if self.cache.contains(candidate):
                # Already evaluated: zero oracle cost, but it DOES count
                # toward patience — repeated hits mean the improving
                # neighborhood of the current anchor is exhausted, which is
                # exactly when a restart should fire.
                steps_since_improve += 1
                continue
            self.cache.add(candidate)

            found_, inst = self.evaluate(candidate)
            if found_:
                best = candidate
                actual = candidate
                result = inst
                steps_since_improve = 0
                restarts_without_improve = 0
                self.logger.info("============> (hill) size: " + str(len(actual)))
            else:
                # Patience counts evaluated (oracle-paid) candidates, so each
                # window really tests `patience` informative neighbors.
                steps_since_improve += 1

        if self.oracle.predict(result) == self.oracle.predict(self.G):
            self.logger.info("ERROR, returning non ctf ")
        return result

    def evaluate(self, solution):
        new_data = np.copy(self.G.data)
        self._disturb(new_data, self.G.directed, solution)

        if self.attributed and self.manip_attr:
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
            if not self.attributed and self.recompute_features:
                self.dataset.manipulate(new_g)
            if self.M.classify(new_g):
                return (True, new_g)

        return (False, None)

    def _disturb(self, data, directed, solution):
        for i in solution:
            (n1, n2) = self.labels[i]
            data[n1, n2] = (data[n1, n2] + 1) % 2
            if not directed:
                data[n2, n1] = (data[n2, n1] + 1) % 2

    def _random_subset(self, solution, size):
        """Pick a random subset of exactly ``size`` from ``solution``.

        Used only for the random restart step (textbook RRHC); the inner
        loop itself does not call this."""
        if size > len(solution):
            return set(solution)
        return set(random.sample(list(solution), size))

    def write(self):
        pass

    def read(self):
        pass
