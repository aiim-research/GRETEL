import numpy as np

from src.dataset.manipulators.base import BaseManipulator


class OracleView(BaseManipulator):
    """Keep only what the GCN oracles read: node features, structure, edge weights.

    Edge features become GraphInstance's default (one column of ones) and graph
    features are dropped. For molecules these are the RDKit bond types and the
    SMILES/RDKit object, which never reach the GCN. The explained graph is then
    the one the oracle uses, the LLM prompts carry no SMILES of the original
    molecule, and the explainers' candidates (built without edge or graph
    features) are not reported as unsupported changes.
    """

    def process(self):
        self.dataset.edge_features_map = {}
        self.dataset.graph_features_map = {}
        for instance in self.dataset.instances:
            self.process_instance(instance)

    def process_instance(self, instance):
        instance.edge_features = np.ones((np.count_nonzero(instance.data), 1), dtype=np.float32)
        instance.graph_features = None
