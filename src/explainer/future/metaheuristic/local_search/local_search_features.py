"""BLS-F: Bounded Local Search over edge edits and node-feature substitutions (protocol v2).

The BLS of the first paper edits only edges, on the input's own node features (and, with ``attributed``, its
global attribute methods). BLS-F also edits node features, locally and towards a reference counterfactual:
it reads the node features of the seed every minimizer gets (the full projection of the generator's
candidate in protocol v2: the candidate's edges and node features on the input's nodes; generation stays
decoupled from minimization), and every node whose features differ in the seed contributes one elementary
edit, "node i takes the seed's feature vector" (in a molecule, an atom substitution such as
carbon to nitrogen). The search space is the input's edge pairs plus those substitutions, and the same BLS
skeleton (removal, swap and addition levels, budgets, cache) minimizes their number, so a counterfactual
can trade a few bonds for a few atom substitutions. Each edit counts one: weighing a substitution against an
edge is the cost question of the unified metric, left to it. No global attribute methods.

Edits stay in the input's frame; under the isolated-node policy "delete" the oracle drops the nodes an
edit isolates and GenerateMinimize materializes the counterfactual without them.
"""
import random

import numpy as np

from src.dataset.instances.graph import GraphInstance
from src.explainer.future.metaheuristic.local_search import local_search as bls
from src.utils.comparison import get_edge_differences


class LocalSearch(bls.LocalSearch):

    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['attributed'] = False

    def minimize(self, explaination):
        instance = explaination.input_instance
        seed = explaination.counterfactual_instances[0]
        if len(seed.data) != len(instance.data):
            return seed      # not in the input's frame: nothing this search can shrink
        self.G = instance
        self.N = instance.num_nodes
        self.E = instance.num_edges
        self.EPlus = int((self.N * (self.N - 1)) / 2)
        self.M = bls.BinaryModel(self.oracle, instance)
        self.labels = self.tagger.tag(instance)

        _, diff_matrix = get_edge_differences(self.G, seed)
        coords = [c for c in zip(*np.where(diff_matrix == 1)) if c[0] < c[1]]
        edge_edits = set(self.tagger.get_indices(self.labels, coords))
        # One substitution per node whose features differ in the seed, numbered after the edge pairs.
        x, y = np.asarray(instance.node_features, dtype=float), np.asarray(seed.node_features, dtype=float)
        changed = np.nonzero(np.abs(x - y).sum(1) > 0)[0] if x.shape == y.shape else []
        self.feature_targets = {self.EPlus + int(i): (int(i), y[i]) for i in changed}
        actual = sorted(edge_edits | set(self.feature_targets))
        best = actual
        if not actual:
            self.logger.info("Initial solution size is 0")
            return seed
        if self.oracle.predict(seed) == self.oracle.predict(self.G):
            self.logger.info("Generator was incapable of finding counterfactual, returning non ctf")
            return seed
        self.logger.info(f"BLS-F initial solution: {len(edge_edits)} edge edits, {len(self.feature_targets)} substitutions")
        self.cache = bls.FixedSizeCache(capacity=500000)
        return self.get_approximation(actual, best, seed)

    def disturb(self, data, directed, solution):
        super().disturb(data, directed, [i for i in solution if i < self.EPlus])

    def evaluate(self, solution):
        if self._out_of_budget():
            return (False, None)
        new_data = np.copy(self.G.data)
        self.disturb(new_data, self.G.directed, solution)
        features = np.array(self.G.node_features, dtype=float, copy=True)
        for i in solution:
            if i in self.feature_targets:
                node, value = self.feature_targets[i]
                features[node] = value
        self.k += 1
        new_g = GraphInstance(id=self.G.id, label=0, data=new_data, directed=self.G.directed, node_features=features)
        if self.recompute_features:
            self.dataset.manipulate(new_g)
        if self.M.classify(new_g):
            return (True, new_g)
        return (False, None)

    def add_random(self, solution, i):
        available = (set(range(1, self.EPlus)) | set(self.feature_targets)) - set(solution)
        if len(available) < i:
            # a graph with fewer free pairs than the level asks for (small graphs without padding,
            # protocol v2): add them all instead of failing; unreachable on padded datasets
            i = len(available)
        solution.update(random.sample(sorted(available), i))
        return solution
