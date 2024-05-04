# TWITTER-Real-Graph-Partial (https://www.chrsmrrs.com/graphkerneldatasets/TWITTER-Real-Graph-Partial.zip)
import numpy as np

from os.path import join

from src.dataset.generators.base import Generator
from src.dataset.instances.graph import GraphInstance

# Python
from collections import defaultdict


class TwitterRGP(Generator):

    def init(self):
            base_path = self.local_config['parameters']['data_dir']
            self._adj_file_path = join(base_path, 'TWITTER-Real-Graph-Partial_A.txt')
            self._graph_ind_file_path = join(base_path, 'TWITTER-Real-Graph-Partial_graph_indicator.txt')
            self._graph_labels_file_path = join(base_path, 'TWITTER-Real-Graph-Partial_graph_labels.txt')
            self._edge_att_file_path = join(base_path, 'TWITTER-Real-Graph-Partial_edge_attributes.txt')
            self._node_labels_file_paht = join(base_path, 'TWITTER-Real-Graph-Partial_node_labels.txt')

            self.dataset.node_features_map = {'label': 0}
            self.dataset.graph_features_map = {'label': 0}

            self.generate_dataset()

    def generate_dataset(self):
        if not len(self.dataset.instances):
            self.read_adjacency_matrices()
    
    
    def read_adjacency_matrices(self):
        """
        Reads the dataset from the adjacency matrices
        """

        # Initialize a dictionary to hold the graph nodes
        graph_nodes = defaultdict(list)

        # Reverse mapping for a given node to know its corresponding graph
        node_graph = {}

        """
        Code to know which node correspond to which graph
        """
        # Open the file and read the lines
        with open(self._graph_ind_file_path, 'r') as file:
            lines = file.readlines()

        # Remove newline characters and convert to integers
        lines = [int(line.strip()) for line in lines]

        # Populate the dictionary ({1:[1,2,3,4] ...})
        node_index = 1
        for graph_index in lines:
            graph_nodes[graph_index].append(node_index)
            node_graph[node_index] = graph_index
            node_index += 1


        """
        Create the list of adj matrix filled with zeros
        """

        #List of adj matrix
        adj_matrix = []
        edge_matrix = []
        for key in graph_nodes.keys():
            #Append to the list a adj matrix filled with zeros that has the right shape
            adj_matrix.append(np.zeros((len(graph_nodes[key]), len(graph_nodes[key])),dtype=np.int32))
            edge_matrix.append(np.zeros((len(graph_nodes[key]), len(graph_nodes[key])),dtype=np.float32))
            

        """
        Get every tuple of arcs
        """


        # Open the file and read the lines
        with open(self._adj_file_path, 'r') as file:
            lines = file.readlines()

        # Remove newline characters and convert to integers
        lines = [line.strip() for line in lines]

        # Using a list comprehension to convert each string into a tuple on int
        tuple_list = [tuple(map(int, pair.split(','))) for pair in lines]

        """
        Get every weigths
        """

        # Open the file and read the lines
        with open(self._edge_att_file_path, 'r') as file:
            lines = file.readlines()

        # Remove newline characters and convert to integers
        edge_weights_list = [line.strip() for line in lines]

        """
        Get the right adj matrix for every graph
        """
        counter = 0
        for tuple_item in tuple_list:
            desired_value = tuple_item[0]
            corresponding_graph = node_graph[desired_value]
            #Get all the nodes for the current graph
            current_graph_node_list = graph_nodes[corresponding_graph]
            #Get the right adj matrix in the list for the current graph and change the 0 to 1 if there is an arc
            adj_matrix[corresponding_graph-1][current_graph_node_list.index(tuple_item[0])][current_graph_node_list.index(tuple_item[1])] = 1
            edge_matrix[corresponding_graph-1][current_graph_node_list.index(tuple_item[0])][current_graph_node_list.index(tuple_item[1])] = edge_weights_list[counter]
            counter +=1

        """
        Append the graph into our instances
        """

        # Open the file and read the lines
        with open(self._graph_labels_file_path, 'r') as file:
            lines = file.readlines()

        # Remove newline characters and convert to integers
        label_list = [line.strip() for line in lines]


        with open(self._node_labels_file_paht, 'r') as file:
            lines = file.readlines()

        node_label_list = [line.strip() for line in lines]



        

        #In our dataset, the labels are 1 or -1 but this framework cannot take -1 as a label it's either 0 or 1
        for key, nodes in graph_nodes.items():
            label = 1
            if int(label_list[key-1]) == -1 :
                label = 0
            
            # get node labes
            node_attr = np.array([node_label_list[x-1] for x in nodes])

            self.dataset.instances.append(GraphInstance(id = int(key), 
                                                        label = label, 
                                                        data = adj_matrix[key-1],
                                                        edge_weights=edge_matrix[key-1].flatten(),
                                                        node_features=node_attr,
                                                        graph_features = {'label' : label}))
