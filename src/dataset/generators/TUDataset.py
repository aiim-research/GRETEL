# https://chrsmrrs.github.io/datasets/docs/format/

import numpy as np
import pandas as pd

from os.path import join,exists
from collections import defaultdict

from src.dataset.generators.base import Generator
from src.dataset.instances.graph import GraphInstance


class TUDataset(Generator):
    
    def init(self, dataset_name = ''):
        self.dataset_name = dataset_name
        base_path = self.local_config['parameters']['data_dir']

        a = join(base_path, f'{self.dataset_name}_A.txt')
        graph_indicator = join(base_path, f'{self.dataset_name}_graph_indicator.txt')
        graph_labels = join(base_path, f'{self.dataset_name}_graph_labels.txt')
        node_labels = join(base_path, f'{self.dataset_name}_node_labels.txt')
        edge_labels = join(base_path, f'{self.dataset_name}_edge_labels.txt')
        node_attributes = join(base_path, f'{self.dataset_name}_node_attributes.txt')
        edge_attributes = join(base_path, f'{self.dataset_name}_edge_attributes.txt')
        graph_attributes = join(base_path, f'{self.dataset_name}_graph_attributes.txt')

        self._a_file_path = a if exists(a) else None
        self._graph_indicator_file_path = graph_indicator if exists(graph_indicator) else None
        self._graph_labels_file_path = graph_labels if exists(graph_labels) else None
        self._node_labels_file_path = node_labels if exists(node_labels) else None
        self._edge_labels_file_path = edge_labels if exists(edge_labels) else None
        self._node_attributes_file_path = node_attributes if exists(node_attributes) else None
        self._edge_attributes_file_path = edge_attributes if exists(edge_attributes) else None
        self._graph_attributes_file_path = graph_attributes if exists(graph_attributes) else None

        self.dataset.graph_features_map = {'label': 0}

        self.generate_dataset()

    def generate_dataset(self):
        if not len(self.dataset.instances):
            self.populate()

    def populate(self):
        a = None
        graph_indicator = None
        graph_labels = None
        node_labels = None
        edge_labels = None
        node_attributes = None
        edge_attributes = None
        graph_attributes = None

        with open(self._a_file_path, "r") as f:
            a = [tuple(map(int, pair.split(','))) for pair in f.readlines()]
        
        with open(self._graph_indicator_file_path, "r") as f:
            graph_indicator = [int(v) for v in f.readlines()]

        with open(self._graph_labels_file_path, "r") as f:
            graph_labels = [max(int(v),0) for v in f.readlines()]
        
        # edges
        if self._edge_labels_file_path:
            with open(self._edge_labels_file_path, "r") as f:
                edge_labels = [max(int(v),0) for v in f.readlines()]
            self.dataset.edge_features_map = {'label': 0}
        
        if self._edge_attributes_file_path:
            with open(self._edge_attributes_file_path, "r") as f:
                edge_attributes = [max(float(v),0) for v in f.readlines()]
            
            if not self.dataset.edge_features_map:
                self.dataset.edge_features_map = {'attribute': 0}
            else:
                self.dataset.edge_features_map['attribute'] = 1

        # nodes
        if self._node_labels_file_path:
            with open(self._node_labels_file_path, "r") as f:
                node_labels = [max(int(v),0) for v in f.readlines()]
            self.dataset.node_features_map = {'label': 0}      

        if self._node_attributes_file_path:
            node_attributes = pd.read_csv(self._node_attributes_file_path, header=None).values

            attr_size = len(node_attributes[0])

            if not self.dataset.node_features_map:
                self.dataset.edge_features_map = {f"attribute_{i}":i for i in range(0,attr_size)}
            else:
                self.dataset.edge_features_map.update({f"attribute_{i}":i for i in range(1,attr_size+1)})

        adjs = []
        edlbs = []
        edattr = []
        nodfeat = []

        # Initialize a dictionary to hold the graph nodes
        graph_nodes = defaultdict(list)

        # Reverse mapping for a given node to know its corresponding graph
        node_graph = defaultdict()
        
        node_index = 1
        for graph_index in graph_indicator:
            graph_nodes[graph_index].append(node_index)
            node_graph[node_index] = graph_index
            node_index += 1

        for g in graph_nodes.keys():
            size = len(graph_nodes[g])
            adjs.append(np.zeros((size,size), dtype=np.int32))
            edlbs.append(np.zeros((size,size), dtype=np.float64))
            edattr.append(np.zeros((size,size), dtype=np.float64))

            if self._node_labels_file_path  and self._node_attributes_file_path:
                nodfeat.append(np.zeros((size,size+1), dtype=np.float64))
                nodfeat[-1][:,0] = np.array([node_labels[x-1] for x in graph_nodes[g]])
            elif self._node_labels_file_path:
                nodfeat.append(np.array([node_labels[x-1] for x in graph_nodes[g]]))
            elif self._node_attributes_file_path:
                nodfeat.append(np.zeros((size,size), dtype=np.float64))

        for u,(i, j) in enumerate(a):
            graph = node_graph[i]
            c = len(graph_nodes[graph])
            adjs[graph-1][i % c ,j % c] = 1
            if edge_labels:
                edlbs[graph-1][i % c ,j % c] = edge_labels[u]
            if edge_attributes:
                edattr[graph-1][i % c ,j % c] = edge_attributes[u]
    
        if self._graph_attributes_file_path:
            with open(self._graph_attributes_file_path, "r") as f:
                graph_attributes = [max(float(v),0) for v in f.readlines()]
            self.dataset.graph_features_map['attribute'] = 1
        
        for i in range(0,graph_index):
            id = i + 1
            label = graph_labels[i]
            data = adjs[i]

            # Graph Features
            graph_feat = [graph_labels[i]]
            if graph_attributes:
                graph_feat.append(graph_attributes[i])
            graph_feat = np.array(graph_feat)

            # Edge Features
            edge_feat = []
            if edge_labels:
                edge_feat.append(edlbs[i])
            if edge_attributes:
                edge_feat.append(edattr[i])
            edge_feat = np.array(edge_feat) if len(edge_feat) else None

            # Node Features
            node_feat = []
            if node_labels:
                node_feat.append(nodfeat[i])
            node_feat = np.array(node_feat) if len(node_feat) else None

            self.dataset.instances.append(GraphInstance(id = id, 
                                                        label = label, 
                                                        data = data,
                                                        graph_features=graph_feat,
                                                        node_features=node_feat,
                                                        edge_features=edge_feat
                                                        ))