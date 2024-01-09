from copy import deepcopy
#from types import NoneType

import networkx as nx
import numpy as np

from src.dataset.instances.base import DataInstance


class GraphInstance(DataInstance):

    def __init__(self, id, label, data, node_features=None, edge_features=None, edge_weights=None, graph_features=None, dataset=None):
        super().__init__(id, label, data, dataset=dataset)
        self.node_features = self.__init_node_features(node_features).astype(np.float32)
        self.edge_features = self.__init_edge_features(edge_features).astype(np.float32)
        self.edge_weights = self.__init_edge_weights(edge_weights).astype(np.float32)
        self.graph_features = graph_features
        self._nx_repr = None

    def __deepcopy__(self, memo):
        # Fields that are being shallow copied
        _dataset = self._dataset

        # Fields that are deep copied
        _new_id = deepcopy(self.id, memo)
        _new_label = deepcopy(self.label, memo)
        _data = deepcopy(self.data, memo)
        _node_features = deepcopy(self.node_features, memo)
        _edge_features = deepcopy(self.edge_features, memo)
        _edge_weights = deepcopy(self.edge_weights, memo)
        _graph_features = deepcopy(self.graph_features, memo)
        return GraphInstance(_new_id, _new_label, _data, _node_features, _edge_features, _edge_weights, _graph_features)

    def get_nx(self):
        if not self._nx_repr:
            self._nx_repr = self._build_nx()
        return deepcopy(self._nx_repr)
    
    def __init_node_features(self, node_features):
        return np.zeros((self.data.shape[0], 1)) if isinstance(node_features, (str, type(None))) else node_features

    def __init_edge_features(self, edge_features):
        edges = np.nonzero(self.data)
        return np.ones((len(edges[0]), 1)) if isinstance(edge_features, (str, type(None))) else edge_features
    
    def __init_edge_weights(self, edge_weights):
        edges = np.nonzero(self.data)
        return np.ones(len(edges[0])) if edge_weights is None else edge_weights
    
    def _build_nx(self):
        nx_repr = nx.from_numpy_array(self.data)
        nx_repr.add_nodes_from([node, {'node_features': self.node_features[node]}] for node in nx_repr.nodes())
        edges = list(nx_repr.edges)
        nx_repr.add_edges_from([(edge[0], edge[1], {'edge_features': self.edge_features[i], 'weight': self.edge_weights[i]}) for i, edge in enumerate(edges)])
        return nx_repr
    
    @property
    def num_edges(self):
        nx_repr = self.get_nx()
        return nx_repr.number_of_edges()
    
    @property
    def num_nodes(self):
        return len(self.data)
            
    def nodes(self):
        return [ i for i in range(self.data.shape[0])]

    def neighbors(self, node):
        return [i for i in self.data[node,:] if i != 0]
    
    def degree(self,node):
        return len(self.neighbors(node))
    
    def degrees(self):
        return [ len(self.neighbors(y)) for y in self.nodes()]