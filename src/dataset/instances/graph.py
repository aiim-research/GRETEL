from copy import deepcopy
#from types import NoneType

import networkx as nx
import numpy as np

from src.dataset.instances.base import DataInstance


class GraphInstance(DataInstance):

    def __init__(self, id, label, data, node_features=None, edge_features=None, edge_weights=None, graph_features=None, dataset=None, directed=False):
        super().__init__(id, label, data, dataset=dataset)
        self.node_features = self.__init_node_features(node_features).astype(np.float32)
        self.edge_features = self.__init_edge_features(edge_features).astype(np.float32)
        self.edge_weights = self.__init_edge_weights(edge_weights).astype(np.float32)
        self.graph_features = graph_features
        self._nx_repr = None
        self.directed = directed


        num_nodes = self.data.shape[0]
        num_edges = np.count_nonzero(self.data)
        assert len(self.node_features) == num_nodes
        assert len(self.edge_features) == num_edges
        assert len(self.edge_weights) == num_edges

    def __deepcopy__(self, memo):
        num_nodes = self.data.shape[0]
        num_edges = np.count_nonzero(self.data)
        assert len(self.node_features) == num_nodes
        assert len(self.edge_features) == num_edges
        assert len(self.edge_weights) == num_edges

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
        _directed = deepcopy(self.directed, memo)

        return GraphInstance(id=_new_id, 
                             label=_new_label, 
                             data=_data, 
                             node_features=_node_features, 
                             edge_features=_edge_features, 
                             edge_weights=_edge_weights, 
                             graph_features=_graph_features, 
                             directed=_directed,
                             dataset=_dataset)

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
    
    @property
    def is_directed(self):
        return self.directed
            
    def nodes(self):
        return [ i for i in range(self.data.shape[0])]

    def neighbors(self, node):
        return [i for i in self.data[node,:] if i != 0]
    
    def degree(self,node):
        return len(self.neighbors(node))
    
    def degrees(self):
        return [ len(self.neighbors(y)) for y in self.nodes()]
