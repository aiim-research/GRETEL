from typing import List

import numpy as np

from src.dataset.instances.base import DataInstance
from src.explainer.ensemble.aggregators.nodes.base import NodeFeatureAggregator
from src.utils.utils import pad_features

class AverageAggregator(NodeFeatureAggregator):

    def aggregate(self, nodes: np.array, instances: List[DataInstance]):
        feature_dim = instances[0].node_features.shape[-1]
        avg_node_features = np.zeros((max(nodes)+1, feature_dim))
        for instance in instances:
            # what nodes are incommon with the nodes of the instance
            # I need to know the node indices to update only their feature vectors
            curr_nodes = np.array(list(set(nodes).intersection(set(range(instance.data.shape[0])))))
            m = max(curr_nodes) + 1
            # in case the explainers don't do anything in terms of feature vectors
            # then I need to add zero d-dimensional vectors to the dummy nodes
            instance.node_features = pad_features(instance.node_features, m)            
            # add the instance node features
            avg_node_features[curr_nodes,:] += instance.node_features[curr_nodes,:]
        return avg_node_features / len(instances)
