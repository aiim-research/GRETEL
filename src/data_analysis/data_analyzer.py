from matplotlib.pyplot import table
from src.dataset.dataset_factory import DatasetFactory

import os
import jsonpickle
import numpy as np
from sklearn import metrics
import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt
import statistics
import sys

from src.core.factory_base import get_instance_kvargs

class DataAnalyzer():

    @classmethod
    def get_json_file_paths(cls, folder_path):
        """Given a folder return a list containing the file paths of all json files inside the folder
          or its subfolders"""
        result = []

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".json"):
                    result.append(os.path.join(root, file))

        return result
    

    @classmethod
    def create_aggregated_dataframe(cls, results_folder_path):
        """This method receives a do-pair folder path. This folder is associated to an specific 
        dataset and oracle combination and should contain folders for each of the explainers tested 
        on that do-pair"""
        results_file_paths = cls.get_json_file_paths(results_folder_path)

        mega_dict = {}
        rows = []
        first_iteration = True
        metric_names = []
        
        # Reading the files and creating a dictionaries with aggregated results for each run
        for results_file_uri in results_file_paths:
            with open(results_file_uri, 'r') as results_file_reader:
                results_plain_text = results_file_reader.read()
                results_dict = jsonpickle.decode(results_plain_text)

                # Getting the dataset, oracle and explainer names
                hashed_scope = results_dict['config']['scope']
                hashed_dataset_name = results_dict['hash_ids']['dataset']
                hashed_oracle_name = results_dict['hash_ids']['oracle']
                exp_name = results_dict['hash_ids']['explainer'].split(sep='-')[0]
                hashed_explainer_name = exp_name
                # hashed_explainer_name = results_dict['hash_ids']['explainer']

                # Creating all the necesary levels in the dictionary
                if not hashed_scope in mega_dict:
                    mega_dict[hashed_scope] = {}

                if not hashed_dataset_name in mega_dict[hashed_scope]:
                    mega_dict[hashed_scope][hashed_dataset_name] = {}

                if not hashed_oracle_name in mega_dict[hashed_scope][hashed_dataset_name]:
                    mega_dict[hashed_scope][hashed_dataset_name][hashed_oracle_name] = {}

                if not hashed_explainer_name in mega_dict[hashed_scope][hashed_dataset_name][hashed_oracle_name]:
                    mega_dict[hashed_scope][hashed_dataset_name][hashed_oracle_name][hashed_explainer_name] = []


                # If correctness is among the metrics then get the correctness for each instance
                correctness_cls, correctness_name = cls.resolve_correctness_class_and_name(results_dict)
                if correctness_cls is not None:
                    correctness_vals = [x['value'] for x in results_dict['results'][correctness_cls]]
                else:
                    correctness_vals = None

                # Aggregate the measures
                aggregated_metrics = []
                for m_class, m_value in results_dict['results'].items():
                    metric_name = m_class.split('.')[-1]
                    if first_iteration: # The metric names are only needed the first time
                         metric_names.append(metric_name)

                    metric = get_instance_kvargs(kls=m_class, param={})
                    vals = [x['value'] for x in m_value]
                    agg_values, agg_std = metric.aggregate(vals, correctness_vals)
                    aggregated_metrics.append(agg_values)

                mega_dict[hashed_scope][hashed_dataset_name][hashed_oracle_name][hashed_explainer_name].append(aggregated_metrics)

            first_iteration = False

        # Creating the header of the table
        column_names = ['scope', 'dataset', 'oracle', 'explainer']
        for m_name in metric_names:
            column_names.append(m_name)
            column_names.append(m_name + '-std')

        # Iterating over the dictionary and agregating different runs and folds together
        rows = []
        for scope_name, datasets in mega_dict.items():
            for dataset_name, oracles in datasets.items():
                for oracle_name, explainers in oracles.items():
                    for explainer_name, runs in explainers.items():
                        row = [scope_name, dataset_name, oracle_name, explainer_name]

                        for m in range(len(metric_names)):
                            m_values = [runs[i][m] for i in range(len(runs))]
                            v_mean = np.mean(m_values)
                            v_std = np.std(m_values)
                            row.append(v_mean)
                            row.append(v_std)

                        rows.append(row)

        # Building the dataframe                  
        result = pd.DataFrame(data=rows, columns=column_names)
        return result


    @classmethod
    def resolve_correctness_class_and_name(cls, results_dict):
        for k in results_dict['results'].keys():
            k_low = k.lower()
            if 'correctness' in k_low:
                return k, k.split('.')[-1]
            
        return None
            

    @classmethod
    def get_node_changes(cls, g1, g2):
        """Input: g1 is a data instance and g2 is its counterfactual instance
           Output: (common nodes, added nodes, removed nodes)"""

        n_g1 = g1.data.shape[0]
        n_g2 = g2.data.shape[0]

        common_nodes_num = min(n_g1, n_g2)
        common_nodes = list(range(0, common_nodes_num))

        if n_g1 == n_g2: # Both graphs have the same size
            removed_nodes = []
            added_nodes = []
        elif n_g1 > n_g2: # G1 has more nodes than G2
            removed_nodes = list(range(common_nodes_num, n_g1))
            added_nodes = []
        else: # G2 has more nodes than G1
            removed_nodes = []
            added_nodes = list(range(common_nodes_num, n_g2))

        return common_nodes, added_nodes, removed_nodes
    

    @classmethod
    def get_edge_changes(cls, g1, g2, directed=False):
        """Returns (common_edges_list, added_edges_list, removed_edges_list)"""
        g1_A = g1.data
        g2_A = g2.data
        n_g1 = g1_A.shape[0]
        n_g2 = g2_A.shape[0]
        common_nodes_num = min(n_g1, n_g2)

        common_edges = []
        added_edges = []
        removed_edges = []

        if directed: # If the graphs are directed

            # Check the edges between the nodes common to both graphs
            for i in range(common_nodes_num):
                for j in range(common_nodes_num):
                    if g1_A[i,j] == 1 and g2_A[i,j] == 1:
                        common_edges.append((i,j))
                    elif g1_A[i,j] == 0 and g2_A[i,j] == 1:
                        added_edges.append((i,j))
                    elif g1_A[i,j] == 1 and g2_A[i,j] == 0:
                        removed_edges.append((i,j))

            # If g2 removed nodes from g1 then all the edges from those nodes were removed
            if n_g1 > n_g2:
                for i in range(common_nodes_num, n_g1):
                    for j in range(0, i):
                        if g1_A[i,j] == 1:
                            removed_edges.append((i,j))

                for j in range(common_nodes_num, n_g1):
                    for i in range(0, j):
                        if g1_A[i,j] == 1:
                            removed_edges.append((i,j))

            # If g2 added nodes then all the edges to those nodes were added
            elif n_g2 > n_g1:
                for i in range(common_nodes_num, n_g2):
                    for j in range(0, j):
                        if g2_A[i,j] == 1:
                            added_edges.append((i,j))
                
                for j in range(common_nodes_num, n_g2):
                    for i in range(0, j):
                        if g2_A[i,j] == 1:
                            added_edges.append((i,j))

        else: # The graph is undirected, so we only iterate over half the graph

            # Check the edges between the nodes common to both graphs
            for i in range(common_nodes_num):
                for j in range(i, common_nodes_num):
                    if g1_A[i,j] == 1 and g2_A[i,j] == 1:
                        common_edges.append((i,j))
                    elif g1_A[i,j] == 0 and g2_A[i,j] == 1:
                        added_edges.append((i,j))
                    elif g1_A[i,j] == 1 and g2_A[i,j] == 0:
                        removed_edges.append((i,j))

            # If g2 removed nodes from g1 then all the edges from those nodes were removed
            if n_g1 > n_g2:
                for j in range(common_nodes_num, n_g1):
                    for i in range(0, j):
                        if g1_A[i,j] == 1:
                            removed_edges.append((i,j))

            # If g2 added nodes then all the edges to those nodes were added
            elif n_g2 > n_g1:
                for j in range(common_nodes_num, n_g2):
                    for i in range(0, j):
                        if g2_A[i,j] == 1:
                            added_edges.append((i,j))

        return common_edges, added_edges, removed_edges
    

    @classmethod
    def get_cf_changes(cls, og_inst, cf_inst, directed=False):
        common_nodes, added_nodes, removed_nodes = cls.get_node_changes(og_inst, cf_inst)
        common_edges, added_edges, removed_edges = cls.get_edge_changes(og_inst, cf_inst, directed)

        return {'common nodes': common_nodes, 
                'common edges': common_edges, 
                'added nodes': added_nodes, 
                'added edges': added_edges,
                'removed nodes': removed_nodes,
                'removed edges': removed_edges}


    @classmethod
    def get_nx_graph(cls, graph_instance):
        # This method exist in case we move the networkx transformation outside of the instances in the future
        return graph_instance.get_nx()
    

    @classmethod
    def draw_graph(cls, data_instance, position=None, img_store_address=None):

        G = cls.get_nx_graph(data_instance)

        if position is None:
            layout = nx.spring_layout
            position = layout(G)

        edge_colors = ['cyan' for u, v in G.edges()]
        node_colors = ['cyan' for node in G.nodes()]

        nx.draw_networkx(G=G, pos=position, node_color=node_colors, edge_color=edge_colors, with_labels=True)

        if img_store_address:
            plt.savefig(img_store_address, format='svg')

        plt.show(block=False)

        # After showing the graph returns the position used, so can be re-used later
        return position


    @classmethod
    def draw_counterfactual_actions(cls, 
                                    og_instance, 
                                    cf_instance, 
                                    position=None, 
                                    img_store_address=None):
        
        # In case a position is not provided
        if position is None:
            layout = nx.spring_layout
            position = layout(cls.get_nx_graph(og_instance))
        
        changes = cls.get_cf_changes(og_instance, cf_instance, False)

        edges_shared = changes['common edges']
        edges_added = changes['added edges']
        edges_deleted = changes['removed edges']
        nodes_shared = changes['common nodes']
        nodes_added = changes['added nodes']
        nodes_deleted = changes['removed nodes']

        # Create a new Network object
        G = nx.Graph()

        # Add shared nodes and edges in grey
        for node in nodes_shared:
            G.add_node(node, color='cyan')
        for edge in edges_shared:
            G.add_edge(*edge, color='cyan')

        # Add deleted nodes and edges in red
        for node in nodes_deleted:
            G.add_node(node, color='red')
        for edge in edges_deleted:
            G.add_edge(*edge, color='red')

        # Add added nodes and edges in green color
        for node in nodes_added:
            G.add_node(node, color='green')
        for edge in edges_added:
            G.add_edge(*edge, color='green')

        edge_colors = [G[u][v]['color'] for u, v in G.edges()]
        node_colors = [G.nodes[node]['color'] for node in G.nodes()]
        
        nx.draw_networkx(G=G, pos=position, node_color=node_colors, edge_color=edge_colors, with_labels=True)

        if img_store_address:
            plt.savefig(img_store_address, format='svg')

        plt.show(block=False)

       