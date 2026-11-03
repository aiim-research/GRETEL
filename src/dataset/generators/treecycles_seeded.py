"""Tree-Cycles generator with a seed and an idempotent generation.

Same graphs as TreeCyclesRand (a random tree, plus cycles hanging off it for
half of the instances), with two differences that matter when the dataset
comes back from the cache:

- The dataset cache stores splits and metadata but not the instances, which
  are regenerated at every load. Dataset.read() also runs the generator's
  init() twice, once through the constructor and once explicitly. Without a
  guard the second run appends a second copy of every instance; without a
  seed the regenerated graphs are not the ones the cached splits were
  computed on. Here generation is skipped when instances already exist, and
  it is driven by a local random state, so the same config always yields the
  same graphs and the global numpy RNG is left untouched.

Parameters (all hashed into the dataset name):
- num_instances, num_nodes_per_instance, ratio_nodes_in_cycles: as in
  TreeCyclesRand
- seed: integer seed of the generation (default 0)
"""
import networkx as nx
import numpy as np

from src.dataset.generators.treecycles_rand import TreeCyclesRand
from src.dataset.instances.graph import GraphInstance


def generate_tree_cycles(num_instances, num_nodes_per_instance, ratio_nodes_in_cycles, seed):
    """The (adjacency, label) pairs of a Tree-Cycles dataset, as a list.

    Pure function so the generation can be tested without a Dataset. Label 1
    means the graph contains cycles, 0 that it is a tree."""
    rng = np.random.RandomState(seed)
    graphs = []
    for _ in range(num_instances):
        has_cycles = rng.randint(0, 2)
        if has_cycles:
            cycles = []
            budget = int(ratio_nodes_in_cycles * num_nodes_per_instance)
            left = num_nodes_per_instance - budget
            while budget > 2:
                num_nodes = rng.randint(3, budget + 1)
                cycles.append(nx.cycle_graph(num_nodes))
                budget -= num_nodes
            left += budget
            adj = _join_graphs_as_adj(nx.random_tree(n=left, seed=rng), cycles, rng)
            graphs.append((adj, 1))
        else:
            tree = nx.random_tree(n=num_nodes_per_instance, seed=rng)
            graphs.append((nx.to_numpy_array(tree), 0))
    return graphs


def _join_graphs_as_adj(base, others, rng):
    """Attach every graph in others to base with one random edge each."""
    A = nx.to_numpy_array(base)
    num_base = len(A)
    for other in others:
        Ao = nx.to_numpy_array(other)
        t_node = rng.randint(0, num_base)
        s_node = len(A) + rng.randint(0, len(Ao))
        A = np.block([[A, np.zeros((len(A), len(Ao)))], [np.zeros((len(Ao), len(A))), Ao]])
        A[t_node, s_node] = 1
        A[s_node, t_node] = 1
    return A


class TreeCyclesSeeded(TreeCyclesRand):

    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters'].setdefault('seed', 0)

    def init(self):
        self.seed = self.local_config['parameters']['seed']
        super().init()

    def generate_dataset(self):
        if self.dataset.instances:
            self.context.logger.info(f'{len(self.dataset.instances)} instances already present, not regenerating')
            return
        graphs = generate_tree_cycles(self.num_instances, self.num_nodes_per_instance,
                                      self.ratio_nodes_in_cycles, self.seed)
        for i, (adj, label) in enumerate(graphs):
            self.dataset.instances.append(GraphInstance(id=i, data=adj, label=label))
        self.context.logger.info(f'Generated {len(graphs)} Tree-Cycles instances with seed {self.seed}')
