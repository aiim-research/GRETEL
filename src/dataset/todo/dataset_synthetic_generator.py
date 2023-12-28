from src.dataset.instances.base import DataInstance
from src.dataset.dataset_base import Dataset

import networkx as nx
import numpy as np

class Synthetic_Data(Dataset):

    def __init__(self, id, config_dict=None) -> None:
        super().__init__(id, config_dict)
        self.instances = []
        
    def __str__(self):
        return self._name#self.__class__.__name__

    def create_tree(self, tree_size, role_label=0, create_using=None):
        """Builds a random tree graph
        -------------
        INPUT:
        tree_size   :    int the number of nodes in the tree
        role_label  :    int the label to be assigned to the nodes of the graph
        -------------
        OUTPUT:
        graph       :    an undirected acyclic graph
        node_labels :    dict containing the role labels of the nodes
        edge_labels :    dict containing the role labels of the edges
        """
        graph = nx.random_tree(n=tree_size)

        # Creating the dictionary containing the node labels 
        node_labels = {}
        for n in graph.nodes:
            node_labels[n] = role_label

        # Creating the dictionary containing the edge labels
        edge_labels = {}
        for e in graph.edges:
            edge_labels[e] = role_label

        return graph, node_labels, edge_labels


    def create_cycle(self, start, cycle_size, role_label=1):
        """Builds a cycle graph, with index of nodes starting at start
        and role_ids at role_start
        -------------
        INPUT:
        start       :    int starting index for the shape
        role_label  :    int the role label for the nodes in the shape
        -------------
        OUTPUT:
        graph       :    a house shape graph, with ids beginning at start
        role_labels :    list of the roles of the nodes
        """

        # Creating an empty graph and adding the nodes
        graph = nx.Graph()
        graph.add_nodes_from(range(start, start + cycle_size))

        # Adding the edges  of the graph
        for i in range(cycle_size - 1):
            graph.add_edges_from([(start + i, start + i + 1)])

        graph.add_edges_from([(start + cycle_size - 1, start)])
        
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


    def join_graph(self, G1, G2, num_add_edges, G1_node_labels, G2_node_labels, G1_edge_labels, G2_edge_labels):
        """ Join two graphs along matching nodes, then add edges between the nodes of both subgraphs.
        -------------
        INPUT:
            G1, G2        : Networkx graphs to be joined.
            num_add_edges : number of edges to be added connecting the two subgraphs.
            G1_node_labels: The role labels of the nodes of G1
            G2_node_labels: The role labels of the nodes of G2
            G1_edge_labels: The role labels of the edges of G1
            G2_edge_labels: The role labels of the edges of G2
        -------------
        OUTPUT:
            merged_graph      : A new graph, result of merging and perturbing G1 and G2.
            merged_node_labels: the node labels of the combined graph
            merged_edge_labels: the edge labels of the combined graph
        """

        # Joining the two graphs
        G_result = nx.compose(G1, G2)

        # Merging the labels of G2 into those of G1
        result_node_labels = {**G1_node_labels, **G2_node_labels}
        result_edge_labels = {**G1_edge_labels, **G2_edge_labels}
        
        # Randomly adding edges between the nodes of the two subgraphs
        edge_cnt = 0
        while edge_cnt < num_add_edges:
            # Choosing the two node to connect
            node_1 = np.random.choice(G1.nodes())
            node_2 = np.random.choice(G2.nodes())
            
            # Creating an edge between the nodes
            G_result.add_edge(node_1, node_2)
            result_edge_labels[(node_1, node_2)] = 0

            edge_cnt += 1

        return G_result, result_node_labels, result_edge_labels


    def create_tree_cycles_graph(self, n_nodes_tree: int, cycle_size: int, n_cycles: int, create_using=None):
        """ Generate a graph composed by a base tree with added cycles connected to the tree by one edge
        -------------
        INPUT:
            n_nodes_tree  : number of nodes of the base tree graph
            cycle_size    : number of nodes in each cycle
            n_cycles      : the number of cycle graphs to add to the base tree
            create_using  : a graph generator (can be used to create a directed tree)
        -------------
        OUTPUT:
            tree-cycle_graph: A base tree graph with some cycle subgraphs connected to it.
            tree-cycle_node_labels: the node labels of the tree-cycle graph
            tree-cycle_edge_labels: the edge labels of the tree-cycle graph
        """

        # Generate the base tree graph
        result_graph, result_node_labels, result_edge_labels = self.create_tree(tree_size=n_nodes_tree,
        role_label=0, create_using=create_using)

        # Generate the cycle graphs and connect them to the tree
        for c in range(0, n_cycles):
            c_graph, c_node_labels, c_edge_labels = self.create_cycle(start=len(result_node_labels), 
            cycle_size=cycle_size, role_label=1)

            result_graph, result_node_labels, result_edge_labels = self.join_graph(G1=result_graph, G2=c_graph, 
                num_add_edges=1, G1_node_labels=result_node_labels, G2_node_labels=c_node_labels,
                G1_edge_labels=result_edge_labels, G2_edge_labels=c_edge_labels)

        # Return the tree-cycle graph
        return result_graph, result_node_labels, result_edge_labels


    def generate_tree_cycles_dataset(self, n_instances=300, n_total=300, n_in_cycles=200, nodes_per_cycle=None, number_of_cycles=None):
        """Generate a dataset of graphs composed by a base tree with added cycles connected to the tree
        by one edge each.
        -------------
        INPUT:
            n_instances : the number of instances in the dataset 
            n_total     : the total number of nodes in each graph
            n_in_cycles : the maximum number of nodes in the cycle subgraphs (the graphs can contain less)

        -------------
        OUTPUT:
            list of data instances: each data instance is a dictionary contains a graph, graph_name,
            graph_label, node_labels, edge_labels, and minimum_counterfactual_distance
        """

        if nodes_per_cycle is not None and number_of_cycles is not None:
            self._name = (f'tree-cycles_instances-{n_instances}_nodes_per_inst-{n_total}_nodes_per_cycle-{nodes_per_cycle}_number_of_cycles-{number_of_cycles}')
        else:
            self._name = ('tree-cycles_instances-'+ str(n_instances) + '_nodes_per_inst-' + str(n_total) +
                            '_nodes_in_cycles-' + str(n_in_cycles))

        # Creating the empty list of instances
        result = []

        for i in range(0, n_instances):
            # Randomly determine if the graph is going to contain cycles or just be a tree
            has_cycles = np.random.randint(0,2)

            # If the graph will contain cycles
            if(has_cycles):
                data_instance = DataInstance(id=self._instance_id_counter)
                self._instance_id_counter +=1

                if nodes_per_cycle is not None and number_of_cycles is not None:
                    n_cycles = number_of_cycles
                    cycle_size = nodes_per_cycle
                else:
                    # Choose randomly the number of nodes in each cycle and the number of cycle subgraphs
                    n_cycles = np.random.randint(low=1, high=66)
                    cycle_size = np.random.randint(low=3, high=20)
                    # Check that the total number is lower than the maximum amount of nodes in cycles
                    while(n_cycles*cycle_size > n_in_cycles):
                        n_cycles = np.random.randint(low=1, high=66)
                        cycle_size = np.random.randint(low=3, high=20)

                # Calculating how many nodes have to contain the tree to maintain the total
                # number of nodes per instance
                n_tree = n_total - (n_cycles*cycle_size)

                # Creating a name for the instance
                tc_name = 'g' + str(i) + '_t' + str(n_tree) + '_cs' + str(cycle_size) + '_cn' + str(n_cycles)

                # Creating the tree-cycles graph
                tc_graph, tc_node_labels, tc_edge_labels = self.create_tree_cycles_graph(n_nodes_tree=n_tree,
                cycle_size=cycle_size, n_cycles=n_cycles)

                # Creating the instance
                data_instance.graph = tc_graph
                data_instance.node_labels = tc_node_labels
                data_instance.edge_labels = tc_edge_labels
                data_instance.graph_label = 1
                data_instance.minimum_counterfactual_distance = n_cycles
                data_instance.name = tc_name

                data_instance.id = self._instance_id_counter
                self._instance_id_counter += 1

                result.append(data_instance)

            else:
                data_instance = DataInstance(id=self._instance_id_counter)
                self._instance_id_counter +=1

                # Generating a random tree containing all the nodes of the instance
                tc_graph, tc_node_labels, tc_edge_labels = self.create_tree(tree_size=n_total, role_label=0)
                tc_name = 'g' + str(i) + '_t' + str(n_total) + '_cs' + str(0) + '_cn' + str(0)

                # Creating the instance
                data_instance.graph = tc_graph
                data_instance.node_labels = tc_node_labels
                data_instance.edge_labels = tc_edge_labels
                data_instance.graph_label = 0
                data_instance.minimum_counterfactual_distance = 1
                data_instance.name = tc_name

                result.append(data_instance)

        # return the set of instances
        self.instances = result


    def generate_tree_cycles_dataset_balanced(self, n_instances_per_class=150, n_total=300, n_in_cycles=200):
        """Generate a dataset of graphs composed by a base tree with added cycles connected to the tree
        by one edge each. This dataset contains the same number of acyclic and non-acyclic graphs
        -------------
        INPUT:
            n_instances_per_class : the number of instances in the dataset elonging to each class
            n_total               : the total number of nodes in each graph
            n_in_cycles           : the maximum number of nodes in the cycle subgraphs (the graphs can contain less)

        -------------
        OUTPUT:
            list of data instances: each data instance is a dictionary contains a graph, graph_name,
            graph_label, node_labels, edge_labels, and minimum_counterfactual_distance
        """

        self._name = ('tree-cycles-balanced_instances_per_class-'+ str(n_instances_per_class) + 
                        '_nodes_per_inst-' + str(n_total) + '_nodes_in_cycles-' + str(n_in_cycles))

        # Creating the empty list of instances
        result = []

        # Creating the graphs with cycles
        for i in range(0, n_instances_per_class):
            # If the graph will contain cycles
                data_instance = DataInstance(id=self._instance_id_counter)
                self._instance_id_counter +=1

                # Choose randomly the number of nodes in each cycle and the number of cycle subgraphs
                n_cycles = np.random.randint(low=1, high=66)
                cycle_size = np.random.randint(low=3, high=20)
                # Check that the total number is lower than the maximum amount of nodes in cycles
                while(n_cycles*cycle_size > n_in_cycles):
                    n_cycles = np.random.randint(low=1, high=66)
                    cycle_size = np.random.randint(low=3, high=20)

                # Calculating how many nodes have to contain the tree to maintain the total
                # number of nodes per instance
                n_tree = n_total - (n_cycles*cycle_size)

                # Creating a name for the instance
                tc_name = 'g' + str(i) + '_t' + str(n_tree) + '_cs' + str(cycle_size) + '_cn' + str(n_cycles)

                # Creating the tree-cycles graph
                tc_graph, tc_node_labels, tc_edge_labels = self.create_tree_cycles_graph(n_nodes_tree=n_tree,
                cycle_size=cycle_size, n_cycles=n_cycles)

                # Creating the instance
                data_instance.graph = tc_graph
                data_instance.node_labels = tc_node_labels
                data_instance.edge_labels = tc_edge_labels
                data_instance.graph_label = 1
                data_instance.minimum_counterfactual_distance = n_cycles
                data_instance.name = tc_name

                result.append(data_instance)

        # Creating the tree graphs
        for i in range(n_instances_per_class, 2*n_instances_per_class):
            data_instance = DataInstance(id=self._instance_id_counter)
            self._instance_id_counter +=1

            # Generating a random tree containing all the nodes of the instance
            tc_graph, tc_node_labels, tc_edge_labels = self.create_tree(tree_size=n_total, role_label=0)
            tc_name = 'g' + str(i) + '_t' + str(n_total) + '_cs' + str(0) + '_cn' + str(0)

            # Creating the instance
            data_instance.graph = tc_graph
            data_instance.node_labels = tc_node_labels
            data_instance.edge_labels = tc_edge_labels
            data_instance.graph_label = 0
            data_instance.minimum_counterfactual_distance = 1
            data_instance.name = tc_name

            result.append(data_instance)

        # return the set of instances
        self.instances = result


    def generate_dataset_dummy(self, n_instances_per_class=150, n_total=300, n_in_cycles=200):
        """Generate a dataset of graphs composed by a base tree with added cycles connected to the tree
        by one edge each. This dataset contains just two instanes repeated one is a tree and the other one
        is a base tree with cyce subgraphs connected to it
        -------------
        INPUT:
            n_instances_per_class : the number of instances in the dataset elonging to each class
            n_total               : the total number of nodes in each graph
            n_in_cycles           : the maximum number of nodes in the cycle subgraphs (the graphs can contain less)

        -------------
        OUTPUT:
            list of data instances: each data instance is a dictionary contains a graph, graph_name,
            graph_label, node_labels, edge_labels, and minimum_counterfactual_distance
        """

        self._name = ('tree-cycles-dummy_instances_per_class-'+ str(n_instances_per_class) + 
                        '_nodes_per_inst-' + str(n_total) + '_nodes_in_cycles-' + str(n_in_cycles))

        # Creating the empty list of instances
        result = []

        # Creating the graphs with cycles
        # Choose randomly the number of nodes in each cycle and the number of cycle subgraphs
        n_cycles = np.random.randint(low=1, high=66)
        cycle_size = np.random.randint(low=3, high=20)
        # Check that the total number is lower than the maximum amount of nodes in cycles
        while(n_cycles*cycle_size > n_in_cycles):
            n_cycles = np.random.randint(low=1, high=66)
            cycle_size = np.random.randint(low=3, high=20)

        # Calculating how many nodes have to contain the tree to maintain the total
        # number of nodes per instance
        n_tree = n_total - (n_cycles*cycle_size)

        # Creating the tree-cycles graph
        tc_graph, tc_node_labels, tc_edge_labels = self.create_tree_cycles_graph(n_nodes_tree=n_tree,
        cycle_size=cycle_size, n_cycles=n_cycles)

        for i in range(0, n_instances_per_class):
            # If the graph will contain cycles
                data_instance = DataInstance(id=self._instance_id_counter)
                self._instance_id_counter +=1

                # Creating a name for the instance
                tc_name = 'g' + str(i) + '_t' + str(n_tree) + '_cs' + str(cycle_size) + '_cn' + str(n_cycles)

                # Creating the instance
                data_instance.graph = tc_graph
                data_instance.node_labels = tc_node_labels
                data_instance.edge_labels = tc_edge_labels
                data_instance.graph_label = 1
                data_instance.minimum_counterfactual_distance = n_cycles
                data_instance.name = tc_name

                result.append(data_instance)

        # Creating the tree graphs
        # Generating a random tree containing all the nodes of the instance
        t_graph, t_node_labels, t_edge_labels = self.create_tree(tree_size=n_total, role_label=0)

        for i in range(n_instances_per_class, 2*n_instances_per_class):
            data_instance = DataInstance(id=self._instance_id_counter)
            self._instance_id_counter +=1

            t_name = 'g' + str(i) + '_t' + str(n_total) + '_cs' + str(0) + '_cn' + str(0)

            # Creating the instance
            data_instance.graph = t_graph
            data_instance.node_labels = t_node_labels
            data_instance.edge_labels = t_edge_labels
            data_instance.graph_label = 0
            data_instance.minimum_counterfactual_distance = 1
            data_instance.name = t_name

            result.append(data_instance)

        # return the set of instances
        self.instances = result


    def create_infinity(self, start, role_label=1):
            """Builds an infinity-shape graph, with index of nodes starting at start
            and role_ids is the label of the nodes belonging to the shape
            -------------
            INPUT:
            start       :    int starting index for the shape
            role_label  :    int the role label for the nodes in the shape
            -------------
            OUTPUT:
            graph       :    a house shape graph, with ids beginning at start
            role_labels :    list of the roles of the nodes
            edge_labels :    list of the roles of the edges
            """

            # Creating an empty graph and adding the nodes
            graph = nx.Graph()
            graph.add_nodes_from(range(start, start + 5))

            # Adding the edges  of the graph
            # First triangle
            graph.add_edge(start, start + 1)
            graph.add_edge(start + 1, start + 2)
            graph.add_edge(start + 2, start)
            # Second triangle
            graph.add_edge(start+2, start+3)
            graph.add_edge(start+3, start+4)
            graph.add_edge(start+4, start+2)
            
            
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

    
    def create_broken_infinity(self, start, role_label=1):
            """Builds an infinity-shape (but missing one edge) graph, with index of nodes starting at start
            and role_ids is the label of the nodes and edges belonging to the shape
            -------------
            INPUT:
            start       :    int starting index for the shape
            role_label  :    int the role label for the nodes in the shape
            -------------
            OUTPUT:
            graph       :    a house shape graph, with ids beginning at start
            role_labels :    list of the roles of the nodes
            edge_labels :    list of the roles of the edges
            """

            # Creating an empty graph and adding the nodes
            graph = nx.Graph()
            graph.add_nodes_from(range(start, start + 5))

            # Adding the edges  of the graph
            # First triangle
            graph.add_edge(start, start + 1)
            graph.add_edge(start + 1, start + 2)
            graph.add_edge(start + 2, start)
            # Second triangle (broken one)
            graph.add_edge(start+2, start+3)
            graph.add_edge(start+4, start+2)

            # Missing edge
            # graph.add_edge(start+3, start+4)
            
            
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


    def create_tree_infinity_graph(self, n_nodes_tree: int, n_infinities: int, n_broken_infinities: int, create_using=None):
        """ Generate a graph composed by a base tree with added cycles connected to the tree by one edge
        -------------
        INPUT:
            n_nodes_tree       : number of nodes of the base tree graph
            n_infinities       : number of infinity-shape patterns in the graph
            n_broken_infinities: number of infinity-shape patterns in the graph
            create_using  : a graph generator (can be used to create a directed tree)
        -------------
        OUTPUT:
            tree-infinity_graph      : A base tree graph with some infinity-shape subgraphs connected to it.
            tree-infinity_node_labels: the node labels of the tree-infinity graph
            tree-infinity_edge_labels: the edge labels of the tree-infinity graph
        """

        # Generate the base tree graph
        result_graph, result_node_labels, result_edge_labels = self.create_tree(tree_size=n_nodes_tree,
        role_label=0, create_using=create_using)

        # Generate the infinity-shape graphs and connect them to the tree
        for i in range(0, n_infinities):
            i_graph, i_node_labels, i_edge_labels = self.create_infinity(start=len(result_node_labels), 
                                                                        role_label=1)

            result_graph, result_node_labels, result_edge_labels = self.join_graph(G1=result_graph, 
                                                                                   G2=i_graph, 
                                                                                   num_add_edges=1, 
                                                                                   G1_node_labels=result_node_labels, 
                                                                                   G2_node_labels=i_node_labels,
                                                                                   G1_edge_labels=result_edge_labels, 
                                                                                   G2_edge_labels=i_edge_labels)

        # Generate the broken-infinity-shape graphs and connect them to the tree
        for bi in range(0, n_broken_infinities):
            bi_graph, bi_node_labels, bi_edge_labels = self.create_broken_infinity(start=len(result_node_labels), 
                                                                        role_label=1)

            result_graph, result_node_labels, result_edge_labels = self.join_graph(G1=result_graph, 
                                                                                   G2=bi_graph, 
                                                                                   num_add_edges=1,
                                                                                   G1_node_labels=result_node_labels, 
                                                                                   G2_node_labels=bi_node_labels,
                                                                                   G1_edge_labels=result_edge_labels, 
                                                                                   G2_edge_labels=bi_edge_labels)

        # Return the infinity-cycle graph
        return result_graph, result_node_labels, result_edge_labels


    def generate_tree_infinity_dataset(self, n_instances=300, n_total=300, n_infinities=10, n_broken_infinities=10):
        """Generate a dataset of graphs composed by a base tree with added cycles connected to the tree
        by one edge each.
        -------------
        INPUT:
            n_instances : the number of instances in the dataset 
            n_total     : the total number of nodes in each graph
            n_infinities: the number of infinity patterns
            n_broken_infinities: The number of broken infinity pattern

        -------------
        OUTPUT:
            list of data instances: each data instance is a dictionary contains a graph, graph_name,
            graph_label, node_labels, edge_labels, and minimum_counterfactual_distance
        """

        # Creating the name of the instance with all the info
        self._name = ('tree-infinity_instances-'+ str(n_instances) + '_nodes_per_inst-' + str(n_total) +
                        '_n_infinities-' + str(n_infinities) + '_n_broken_infinities-' 
                        + str(n_broken_infinities))

        # Creating the empty list of instances
        result = []

        for i in range(0, n_instances):
            # Randomly determine if the graph is going to contain infinities or just be a tree
            has_infinities = np.random.randint(0,2)

            # If the graph will contain infinities
            if(has_infinities):
                data_instance = DataInstance(id=self._instance_id_counter)
                self._instance_id_counter +=1

                # Choose randomly the number of infinity patterns
                real_n_infinities = np.random.randint(low=1, high=n_infinities+1)
                real_n_broken_infinities = np.random.randint(low=1, high=n_broken_infinities+1)

                # Calculating how many nodes have to contain the tree to maintain the total
                # number of nodes per instance
                n_tree = n_total - (5*(real_n_infinities + real_n_broken_infinities))

                # Creating a name for the instance
                ti_name = ('g' + str(i) + '_t' + str(n_tree) + '_inf' + str(real_n_infinities) + 
                            '_brkn_inf' + str(real_n_broken_infinities))

                # Creating the tree-infinity graph
                ti_graph, ti_node_labels, ti_edge_labels = self.create_tree_infinity_graph(n_nodes_tree=n_tree,
                n_infinities=real_n_infinities, n_broken_infinities=real_n_broken_infinities)

                # Creating the instance
                data_instance.graph = ti_graph
                data_instance.node_labels = ti_node_labels
                data_instance.edge_labels = ti_edge_labels
                data_instance.graph_label = 1
                data_instance.minimum_counterfactual_distance = real_n_infinities
                data_instance.name = ti_name

                # Assigning and increasing the instance id
                data_instance.id = self._instance_id_counter
                self._instance_id_counter += 1

                result.append(data_instance)

            else:
                data_instance = DataInstance(id=self._instance_id_counter)
                self._instance_id_counter +=1

                # Choose randomly the number of infinity patterns
                real_n_broken_infinities = np.random.randint(low=0, high=n_broken_infinities+1)

                # Calculating how many nodes have to contain the tree to maintain the total
                # number of nodes per instance
                n_tree = n_total - (5*real_n_broken_infinities)

                # Creating a name for the instance
                t_name = ('g' + str(i) + '_t' + str(n_tree) + '_inf' + str(0) + 
                            '_brkn_inf' + str(real_n_broken_infinities))

                # Creating the tree-infinity graph
                t_graph, t_node_labels, t_edge_labels = self.create_tree_infinity_graph(n_nodes_tree=n_tree,
                n_infinities=0, n_broken_infinities=real_n_broken_infinities)

                # Creating the instance
                data_instance.graph = t_graph
                data_instance.node_labels = t_node_labels
                data_instance.edge_labels = t_edge_labels
                data_instance.graph_label = 0
                data_instance.minimum_counterfactual_distance = 1 if real_n_broken_infinities > 0 else 2
                data_instance.name = t_name

                result.append(data_instance)

        # return the set of instances
        self.instances = result