from typing import List

import numpy as np

from src.dataset.instances.base import DataInstance
from src.explainer.ensemble.aggregators.nodes.base import NodeFeatureAggregator

class AverageAggregator(NodeFeatureAggregator):

    def aggregate(self, nodes: np.array, instances: List[DataInstance]):
        feature_dim = instances[0].node_features.shape[-1]
        avg_node_features = np.zeros((max(nodes)+1, feature_dim))
        for instance in instances:
            curr_nodes = np.array(list(set(nodes).intersection(set(range(instance.data.shape[0])))))
            avg_node_features[curr_nodes,:] += instance.node_features[curr_nodes,:]
        return avg_node_features / len(instances)
