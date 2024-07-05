# https://chrsmrrs.github.io/datasets/docs/format/

import os
import numpy as np
import pandas as pd

from os.path import join,exists
from collections import defaultdict

from src.dataset.generators.base import Generator
from src.dataset.instances.graph import GraphInstance

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

GRAPH, EDGE, NODE = 1,2,3

class TUDataset(Generator):

    REPO_URL="https://www.chrsmrrs.com/graphkerneldatasets/"

    def prepare_data(self):
        base_path = os.path.join(self.context.working_store_path,self.dataset_name)
        self.context.logger.info("Dataset Data Path:\t" + base_path)
        if not os.path.exists(base_path):
            self.context.logger.info("Downloading " + self.dataset_name + "...")
            os.makedirs(base_path, exist_ok=True)
            resp = urlopen(self.REPO_URL+self.dataset_name+".zip")
            zipped = ZipFile(BytesIO(resp.read()))
            zipped.extractall(self.context.working_store_path)
            self.context.logger.info("Extracted in " + self.context.working_store_path)

        return base_path
    
    def init(self):
        self.dataset_name = self.local_config['parameters']['alias']
        base_path = self.prepare_data()

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

        self.generate_dataset()

    def generate_dataset(self):
        if not len(self.dataset.instances):
            self.populate()

    def normalize_labels(self, labels):
        original_shape = labels.shape  
        flattened_labels = labels.flatten()  
        sorted_labels = np.unique(sorted(flattened_labels))  

        index_map = {value: index for index, value in enumerate(sorted_labels)}  
        new_labels = np.array([index_map[value] for value in flattened_labels])  
        new_labels = new_labels.reshape(original_shape)  
        
        return new_labels

    def _read_labels_and_attributes(self, label_path, attribute_path, kind):
        labels = None
        attributes = None
        general = None
        features_map  = None

        if label_path:
            labels = pd.read_csv(label_path, header=None).values
            labels = self.normalize_labels(labels)
            #for ind in range(labels.shape[1]):
            #    labels[:,ind]  = labels[:,ind] - min(labels[:,ind])

        if attribute_path:
            attributes = pd.read_csv(attribute_path, header=None).values

        if label_path or attribute_path:
            features_map = {}
            label_count = len(labels[0]) if label_path else 0
            attribute_count = len(attributes[0]) if attribute_path else 0

            #if label_path:
            #    features_map.update({f'label_{i}': i for i in range(label_count)})

            if attribute_path:
                features_map.update({f'attribute_{i}': i for i in range(label_count, label_count + attribute_count)})

            general = np.array([np.append(labels[i], attributes[i]) for i in range(labels.shape[0])]) if label_path and attribute_path else labels if label_path else attributes

        if features_map is not None:
            if kind == GRAPH:
                self.dataset.graph_features_map = features_map
            elif kind == EDGE:
                self.dataset.edge_features_map = features_map
            else:
                self.dataset.node_features_map = features_map

        return general
    
    def populate(self):
        a = None
        graph_indicator = None
        node_attributes = None
        edge_attributes = None
        graph_attributes = None

        with open(self._a_file_path, "r") as f:
            a = [tuple(map(int, pair.split(','))) for pair in f.readlines()]
        
        with open(self._graph_indicator_file_path, "r") as f:
            graph_indicator = [int(v) for v in f.readlines()]

        graph_attributes = self._read_labels_and_attributes(
            self._graph_labels_file_path, 
            self._graph_attributes_file_path,
            GRAPH)
        
        edge_attributes = self._read_labels_and_attributes(
            self._edge_labels_file_path,
            self._edge_attributes_file_path, 
            EDGE)
        
        node_attributes = self._read_labels_and_attributes(
            self._node_labels_file_path, 
            self._node_attributes_file_path, 
            NODE)

        adjs = [np.array([])]
    
        # Initialize a dictionary to hold the graph nodes
        graph_nodes = defaultdict(list)

        # graph edges 
        edgs = defaultdict(list)

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

        for u,(i, j) in enumerate(a, start=1):
            graph = node_graph[i]
            c = len(graph_nodes[graph])
            adjs[graph][i % c ,j % c] = 1
            edgs[graph].append(u)

        for i in graph_nodes.keys():
            id = i
            label = graph_attributes[i-1][0] if graph_attributes is not None else None 
            data = adjs[i]

            graph_feat = graph_attributes[i-1] if graph_attributes is not None else None
            edge_feat = edge_attributes[min(edgs[i])-1:max(edgs[i])] if edge_attributes is not None else None
            node_feat = node_attributes[min(graph_nodes[i])-1:max(graph_nodes[i])] if node_attributes is not None else None

            self.dataset.instances.append(GraphInstance(id = id, 
                                                        label = label, 
                                                        data = data,
                                                        graph_features=graph_feat,
                                                        node_features=node_feat,
                                                        edge_features=edge_feat
                                                        ))