import os
import pickle
from ast import List

import jsonpickle
import networkx as nx
import numpy as np
import torch
from sqlalchemy import true

from src.dataset.data_instance_node import NodeDataInstance
from src.dataset.dataset_base import Dataset


class NodeDataset(Dataset):

    def __init__(self, id, config_dict=None) -> None:
        super().__init__(id, config_dict)
        self.instances: List[NodeDataInstance] = []
        self.name = 'node_dataset'

    def write_data(self, datasets_folder_path):
        # Creating a folder to contain the dataset
        dataset_path = os.path.join(datasets_folder_path, self._name)
        os.makedirs(dataset_path, exist_ok = True)

        # Creating a file to contain the name of the dataset
        with open(os.path.join(dataset_path, 'dataset_name.txt'), 'w') as ds_name_writer:
            ds_name_writer.write(self._name)

        # Creating a file to contain the id of the dataset
        if self.id is not None:
            with open(os.path.join(dataset_path, 'dataset_id.json'), 'w') as ds_id_writer:
                ds_id_writer.write(jsonpickle.encode(self._id))

        if (not len(self.instances) > 0):
            raise ValueError("No data instances to serialize")

        # Creating a file to contain the name (target node) of each instance
        f_gnames = open(os.path.join(dataset_path, 'graph_names.txt'), 'a')

        # same graph data is used for every instance
        graph_data = self.instances[0].graph_data
        
        # save general graph data 
        with open(os.path.join(dataset_path, "graph.sav"), "w") as graph_file:
            graph_file.write(jsonpickle.encode(graph_data))
        
        f_gnames.writelines([f"{i.name}\n" for i in self.instances])
        

    def read_data(self, dataset_path):
        # Reading the name of the dataset from file
        with open(os.path.join(dataset_path, 'dataset_name.txt'), 'r') as ds_name_reader:
            self._name = ds_name_reader.read()

        # Reading the id of the dataset from file
        dataset_id_uri = os.path.join(dataset_path, 'dataset_id.json')
        if os.path.exists(dataset_id_uri):
            with open(dataset_id_uri, 'r') as ds_id_reader:
                str_id = jsonpickle.decode(ds_id_reader.read())
                self._id = str_id

        # Reading the file containing the name of each instance
        f_gnames = open(os.path.join(dataset_path, 'graph_names.txt'), 'r')

        # load general graph data 
        graph_file = open(os.path.join(dataset_path, "graph.sav"), "r")
        graph_data =jsonpickle.decode(graph_file.read())
        graph_file.close()

        graph = self.get_nx_from_edges_list(graph_data)

        result = []
        # Iterate over each instance and load them
        for line in f_gnames.readlines():
            target = int(line)
            inst = NodeDataInstance(name = line, id=target, graph_data=graph_data, target_node=target)
            inst.graph = graph
            inst.node_labels = nx.get_node_attributes(graph, "label")
            result.append(inst)

        f_gnames.close()
        self.instances = result
    
    
    def get_nx_from_edges_list(self, graph_data):
        adj_matrix = torch.Tensor(graph_data["adj"]).squeeze()
        features = torch.Tensor(graph_data["feat"]).squeeze()
        labels = torch.tensor(graph_data["labels"]).squeeze()
        
        graph = nx.Graph()
        n_features = {}
        n_labels = {}

        for node_idx in range(len(adj_matrix)):
            node_adj = adj_matrix[node_idx]

            n_features[node_idx] = features[node_idx]
            n_labels[node_idx] = labels[node_idx].item()
            
            for other_node_idx in range(len(node_adj)):
                if (node_adj[other_node_idx] > 0):
                    graph.add_edge(node_idx, other_node_idx)
            
        nx.set_node_attributes(graph, n_features, "features")
        nx.set_node_attributes(graph, n_labels, "label")

        return graph

    def get_data_vic(self):
        pass
