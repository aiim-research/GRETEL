from src.dataset.instances.base import DataInstance
from src.dataset.dataset_base import Dataset

import networkx as nx
import numpy as np


class TrianglesSquaresDataset(Dataset):

    def __init__(self, id, config_dict=None) -> None:
        super().__init__(id, config_dict)
        self.instances = []

    def create_cycle(self, cycle_size, role_label=1):

        # Creating an empty graph and adding the nodes
        graph = nx.Graph()
        graph.add_nodes_from(range(0, cycle_size))

        # Adding the edges  of the graph
        for i in range(cycle_size - 1):
            graph.add_edges_from([(i, i + 1)])

        graph.add_edges_from([(cycle_size - 1, 0)])
        
        # Creating the dictionary containing the node labels 
        node_labels = {}
        for n in graph.nodes:
            node_labels[n] = role_label

        # Creating the dictionary containing the edge labels
        edge_labels = {}
        for e in graph.edges:
            edge_labels[e] = role_label

        # Returning the cycle graph and the role labels
        return graph, node_labels, edge_labels


    def generate_dataset(self, n_instances):

        self._name = ('triangles-squares_instances-'+ str(n_instances))

        # Creating the empty list of instances
        result = []

        for i in range(0, n_instances):
            # Randomly determine if the graph is going to be a traingle or a square
            is_triangle = np.random.randint(0,2)

            # Creating the instance
            data_instance = DataInstance(id=self._instance_id_counter)
            self._instance_id_counter +=1

            i_name = 'g' + str(i)
            i_graph = None
            i_node_labels = None
            i_edge_labels = None

            # creating the instance properties specific for squares or triangles
            if(is_triangle):
                # Creating the triangle graph
                i_graph, i_node_labels, i_edge_labels = self.create_cycle(cycle_size=3, role_label=1)
                data_instance.graph_label = 1
            else:
                i_graph, i_node_labels, i_edge_labels = self.create_cycle(cycle_size=4, role_label=0)
                data_instance.graph_label = 0  

            # Creating the general instance properties
            data_instance.graph = i_graph
            data_instance.node_labels = i_node_labels
            data_instance.edge_labels = i_edge_labels
            data_instance.minimum_counterfactual_distance = 4
            data_instance.name = i_name

            result.append(data_instance)

        # return the set of instances
        self.instances = result