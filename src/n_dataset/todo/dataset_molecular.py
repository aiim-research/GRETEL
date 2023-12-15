from src.dataset.dataset_base import Dataset
from src.dataset.data_instance_molecular import MolecularDataInstance

from abc import ABC, abstractmethod
from typing import Dict
import os
import ast
import jsonpickle
import networkx as nx
from sklearn.model_selection import KFold
import numpy as np

class MolecularDataSet(Dataset):

    def __init__(self, id, force_fixed_nodes=False, config_dict=None) -> None:
        super().__init__(id, config_dict)
        self.name = 'unnamed_molecular_dataset'
        self._force_fixed_nodes = force_fixed_nodes
        self.max_molecule_len = 0
        self.max_n_atoms = 0

    @property
    def force_fixed_nodes(self):
        return self._force_fixed_nodes

    @property
    def max_n_nodes(self):
        return self.max_molecule_len

    @max_n_nodes.setter
    def max_n_nodes(self, new_val):
        self.max_molecule_len = new_val


    @property
    def n_node_types(self):
        return self.max_n_atoms

    @n_node_types.setter
    def n_node_types(self, new_val):
        self.max_n_atoms = new_val


    def write_data(self, datasets_folder_path, graph_format='edge_list'):
        """Writes the dataset into files inside a given folder
        -------------
        INPUT:
            instances    : the instance in the dataset to write 
            datasets_folder_path : the path to the folder where the dataset are store, 
            a folder will be created inside it to store this dataset

        -------------
        OUTPUT:
            void
        """
        # Creating a folder to contain the dataset
        dataset_path = os.path.join(datasets_folder_path, self._name)
        os.mkdir(dataset_path)

        # Creating a file to contain the name of the dataset
        with open(os.path.join(dataset_path, 'dataset_name.txt'), 'w') as ds_name_writer:
            ds_name_writer.write(self._name)

        # Writing the number of different atoms in the dataset
        with open(os.path.join(dataset_path, 'natoms.json'), 'w') as natoms_writer:
            natoms_writer.write(jsonpickle.encode(self.max_n_atoms))

        # Writing maximum molecule length in the dataset
        with open(os.path.join(dataset_path, 'mlen.json'), 'w') as mlen_writer:
            mlen_writer.write(jsonpickle.encode(self.max_molecule_len))

        # Creating a file to contain the id of the dataset
        if self.id is not None:
            with open(os.path.join(dataset_path, 'dataset_id.json'), 'w') as ds_id_writer:
                ds_id_writer.write(jsonpickle.encode(self._id))

        # Creating a file to contain the name of each instance
        f_gnames = open(os.path.join(dataset_path, 'graph_names.txt'), 'a')

        # Saving each instance
        for i in self.instances:
            # Writing the name of the instance into graph_names.txt
            i_name = i.name
            f_gnames.writelines(i_name + '\n')
            
            # Creating a folder to contain the files associated with the instance
            i_path = os.path.join(dataset_path, i_name)
            os.mkdir(i_path)

            # Writing the molecule in smiles format
            with open(os.path.join(i_path, i_name + '_smiles.json'), 'w') as smiles_writer:
                    smiles_writer.write(i.smiles)

            # Writing the graph
            if (not self._force_fixed_nodes):
                if graph_format == 'edge_list':
                    # Writing the instance graph into edgelist format
                    i_graph_path = os.path.join(i_path, i_name + '_graph.edgelist')
                    nx.write_edgelist(i.graph, i_graph_path)
                elif graph_format == 'adj_matrix':
                    # Writing the instance graph into adj_matrix format
                    i_graph_path = os.path.join(i_path, i_name + '_graph.adjlist')
                    nx.write_multiline_adjlist(i.graph, i_graph_path)
                else:
                    raise ValueError('The chosen graph format is not supported')

            # Writing the node labels into file in json format
            if i.node_labels is not None:
                with open(os.path.join(i_path, i_name + '_node_labels.json'), 'w') as node_labels_writer:
                    node_labels_writer.write(jsonpickle.encode(i.node_labels)) 

            # Writing the edge labels into file in json format
            if i.edge_labels is not None:
                with open(os.path.join(i_path, i_name + '_edge_labels.json'), 'w') as edge_labels_writer:
                    edge_labels_writer.write(jsonpickle.encode(i.edge_labels))

            # Writing the graph label into file in json format
            if i.graph_label is not None:
                with open(os.path.join(i_path, i_name + '_graph_label.json'), 'w') as graph_label_writer:
                    graph_label_writer.write(jsonpickle.encode(i.graph_label))

            # Writing the minimal counterfactual distance into file in json format
            if i.minimum_counterfactual_distance is not None:
                with open(os.path.join(i_path, i_name + '_mcd.json'), 'w') as mcd_writer:
                    mcd_writer.write(jsonpickle.encode(i.minimum_counterfactual_distance))
 
        # Writing the splits into file in json format
        if self.splits is not None:
            with open(os.path.join(i_path, i_name + '_splits.json'), 'w') as split_writer:
                split_writer.write(jsonpickle.encode(self.splits))

        f_gnames.close()


    def read_data(self, dataset_path, graph_format='edge_list'):
        """Reads the dataset from files inside a given folder
        -------------
        INPUT:
            dataset_path : the path to the folder containing the dataset
        -------------
        OUTPUT:
            A list of instances (dictionaries) containing the graphs, labels, and 
            minimum counterfactual distance
        """

        # Reading the name of the dataset from file
        with open(os.path.join(dataset_path, 'dataset_name.txt'), 'r') as ds_name_reader:
            self._name = ds_name_reader.read()

        # Reading the number of different atoms in the dataset from file
        with open(os.path.join(dataset_path, 'natoms.json'), 'r') as natoms_reader:
            str_natoms = jsonpickle.decode(natoms_reader.read())
            self.max_n_atoms = int(str_natoms)

        # Reading the maximum molecule size in the dataset from file
        with open(os.path.join(dataset_path, 'mlen.json'), 'r') as mlen_reader:
            str_mlen = jsonpickle.decode(mlen_reader.read())
            self.max_n_atoms = int(str_mlen)

        # Reading the id of the dataset from file
        dataset_id_uri = os.path.join(dataset_path, 'dataset_id.json')
        if os.path.exists(dataset_id_uri):
            with open(dataset_id_uri, 'r') as ds_id_reader:
                str_id = jsonpickle.decode(ds_id_reader.read())
                self._id = int(str_id)

        # Reading the file containing the name of each instance
        f_gnames = open(os.path.join(dataset_path, 'graph_names.txt'), 'r')

        result = []
        instance_number = 0
        # Iterate over each instance and load them
        for line in f_gnames.readlines():
            inst = MolecularDataInstance(id=instance_number)
            instance_number +=1

            # Getting the instance name and storing the path to the instance folder
            i_name = str.strip(line, '\n')
            inst.name = i_name
            i_path = os.path.join(dataset_path, i_name)

            smiles_uri = os.path.join(i_path, i_name + '_smiles.json')
            with open(smiles_uri, 'r') as smiles_reader: 
                inst.smiles = smiles_reader.read()

            if (not self.force_fixed_nodes):
                if graph_format == 'edge_list':
                    # Reading the graph from the edgelist
                    i_path_graph = os.path.join(i_path, i_name + '_graph.edgelist')
                    # If this line is removed the keys of the nodes are casted to str
                    g = nx.read_edgelist(i_path_graph, nodetype=int)
                    inst.graph = g
                elif graph_format == 'adj_matrix':
                    # Reading the graph from the edgelist
                    i_path_graph = os.path.join(i_path, i_name + '_graph.adjlist')
                    # If this line is removed the keys of the nodes are casted to str
                    g = nx.read_multiline_adjlist(i_path_graph, nodetype=int)
                    inst.graph = g
                else:
                    raise ValueError('The chosen graph format is not supported')
            else:
                inst.molecule_to_graph(True, self.max_n_atoms, self.max_molecule_len)
                    
            # Reading the node labels from json file
            node_labels_uri = os.path.join(i_path, i_name + '_node_labels.json')
            if os.path.exists(node_labels_uri):
                with open(node_labels_uri, 'r') as node_labels_reader: 
                    str_dict = jsonpickle.decode(node_labels_reader.read())
                    node_labels = {}
                    for k, v in str_dict.items():
                        node_labels[ast.literal_eval(k)] = v
                    inst.node_labels = node_labels

            # Reading the edge labels from json file
            edge_labels_uri = os.path.join(i_path, i_name + '_edge_labels.json')
            if os.path.exists(edge_labels_uri):
                with open(edge_labels_uri, 'r') as edge_labels_reader: 
                    str_dict = jsonpickle.decode(edge_labels_reader.read())
                    edge_labels = {}
                    for k, v in str_dict.items():
                        edge_labels[ast.literal_eval(k)] = v
                    inst.edge_labels = edge_labels    

            # Reading the graph label from json file
            graph_label_uri = os.path.join(i_path, i_name + '_graph_label.json')
            if os.path.exists(graph_label_uri):
                with open(graph_label_uri, 'r') as graph_label_reader: 
                    inst.graph_label = jsonpickle.decode(graph_label_reader.read())

            # Reading the minimum counterfactual distance from json file
            mcd_uri = os.path.join(i_path, i_name + '_mcd.json')
            if os.path.exists(mcd_uri):
                with open(mcd_uri, 'r') as mcd_reader: 
                    inst.minimum_counterfactual_distance = int( jsonpickle.decode(mcd_reader.read()) )

            result.append(inst)

        # Reading the splits of the dataset
        splits_uri = os.path.join(i_path, i_name + '_splits.json')
        if os.path.exists(splits_uri):
            with open(splits_uri, 'r') as split_reader: 
                sp = jsonpickle.decode(split_reader.read())
                self.splits = sp

        f_gnames.close()
        self.instances = result