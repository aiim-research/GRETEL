from matplotlib.pyplot import table
from src.dataset.dataset_factory import DatasetFactory

import os
import jsonpickle
import numpy as np
from sklearn import metrics
import pandas as pd
import networkx as nx

import networkx as nx
import matplotlib.pyplot as plt
import statistics
import sys


class DataAnalyzer():

    def __init__(self, stats_folder, output_folder) -> None:
        self.stats_folder = stats_folder
        self.output_folder = output_folder
        self.data_dict = {}

    
    def aggregate_data(self):
        stat_files = self._get_files_list()

        for stat_file_uri in stat_files:
            with open(stat_file_uri, 'r') as stat_file_reader:
                stat_dict = jsonpickle.decode(stat_file_reader.read())

                dataset_name = stat_dict['config']['dataset']['name']
                oracle_name = stat_dict['config']['oracle']['name']
                explainer_name = stat_dict['config']['explainer']['name']

                if dataset_name not in self.data_dict:
                    self.data_dict[dataset_name] = { oracle_name: {explainer_name: [] }}

                elif oracle_name not in self.data_dict[dataset_name]:
                    self.data_dict[dataset_name][oracle_name] = { explainer_name: [] }

                elif explainer_name not in self.data_dict[dataset_name][oracle_name]:
                    self.data_dict[dataset_name][oracle_name][explainer_name] = []

                # getting the data for each metric
                h_mean_list = []
                eps = 0.00000000001
                metrics = {}
                for k, v in stat_dict.items():
                    if k != 'config':

                         # Ignoring instances with correctness 0
                        if k != 'config' and k != 'Correctness' and k != 'Fidelity':
                            v_filtered = [item for item, flag in zip(v, stat_dict['Correctness']) if flag == 1]
                            v_mean = np.mean(v_filtered)
                            metrics[k] = v_mean
                        elif k == 'Correctness' or k == 'Fidelity':
                            v_mean = np.mean(v)
                            metrics[k] = v_mean

                        # v_mean = np.mean(v)
                        # metrics[k] = v_mean

                        # Adding the harmonic mean
                        if (k == 'Correctness' or k == 'Fidelity'):
                            h_mean_list.append((1-v_mean) + sys.float_info.epsilon)
                        if (k == 'Graph_Edit_Distance' or k == 'Sparsity'):
                            h_mean_list.append(v_mean)

                            
                metrics['h_mean'] = statistics.harmonic_mean(h_mean_list)

                

                self.data_dict[dataset_name][oracle_name][explainer_name].append(metrics)


    def aggregate_runs(self):
        # for each dataset
        for dataset in self.data_dict:
            # for each oracle
            for oracle in self.data_dict[dataset]:
                # for each explainer
                for explainer, runs in self.data_dict[dataset][oracle].items():
                    # for each run
                    metrics = {}
                    for run in runs:
                        for metric, value in run.items():
                            if metric not in metrics:
                                metrics[metric] = []
                            metrics[metric].append(value)

                    aggregated_metrics = {}
                    for metric, values in metrics.items():
                        aggregated_metrics[metric] = np.mean(values)
                        aggregated_metrics[metric + '-std'] = np.std(values)

                    self.data_dict[dataset][oracle][explainer] = aggregated_metrics


    def create_tables_by_oracle_dataset(self):
        # iterate over all datasets
        for dataset in self.data_dict:
            # iterate over all oracles
            for oracle in self.data_dict[dataset]:
                # For each dataset-oracle create a table
                data = {'explainer': []}
                for explainer in self.data_dict[dataset][oracle]:
                    data['explainer'].append(explainer)
                    for metric, value in self.data_dict[dataset][oracle][explainer].items():
                        if not metric in data:
                            data[metric] = []
                        data[metric].append(value)

                table = pd.DataFrame(data)
                
                table_path_csv = os.path.join(self.output_folder, dataset + '-' + oracle + '.csv')
                table_path_tex = os.path.join(self.output_folder, dataset + '-' + oracle + '.tex')

                table.to_csv(table_path_csv)
                table.to_latex(table_path_tex)


    def get_datasets_stats(self, dataset_dicts, data_store_path):
        d_fact = DatasetFactory(data_store_path)
        datasets = []

        for d_dict in dataset_dicts:
            ds = d_fact.get_dataset_by_name(d_dict)
            datasets.append(ds)

        data = {'name': [], 
                'instance_number': [], 
                'nodes_mean': [], 
                'nodes_std':[], 
                'edges_mean':[], 
                'edges_std': [], 
                'inst_class_0':[], 
                'inst_class_1':[], 
                'diameter_mean': [],
                'mean_node_degree': []}

        for dts in datasets:
            nodes = np.array([len(i.graph.nodes) for i in dts.instances])
            edges = np.array([len(i.graph.edges) for i in dts.instances])

            class_0_count = 0
            class_1_count = 0
            for i in dts.instances:
                if i.graph_label == 0:
                    class_0_count +=1
                else:
                    class_1_count +=1

            # Calculating mean diamater
            diameter_mean = None
            try:
                diameter_mean = np.mean([nx.diameter(i.graph) for i in dts.instances])
            except:
                diameter_mean = -1

            # Calculating mean degree
            instance_degrees = []
            for i in dts.instances:
                g_degree = np.mean([val for (node, val) in i.graph.degree()])
                instance_degrees.append(g_degree)
            mean_node_degree = np.mean(instance_degrees)


            data['name'].append(dts.name)
            data['instance_number'].append(len(dts.instances))
            data['nodes_mean'].append(np.mean(nodes))
            data['nodes_std'].append(np.std(nodes))
            data['edges_mean'].append(np.mean(edges))
            data['edges_std'].append(np.std(edges))
            data['inst_class_0'].append(class_0_count)
            data['inst_class_1'].append(class_1_count)
            data['diameter_mean'].append(diameter_mean)
            data['mean_node_degree'].append(mean_node_degree)
            

            table = pd.DataFrame(data)
            table_path_csv = os.path.join(self.output_folder, 'dataset_stats.csv')
            table_path_tex = os.path.join(self.output_folder, 'dataset_stats.tex')

            table.to_csv(table_path_csv)
            table.to_latex(table_path_tex)


    def _get_files_list(self):
        # create a list of file and sub directories 
        # names in the given directory 
        oracle_dataset_folders = os.listdir(self.stats_folder)
        result = []
        # Iterate over all the oracle_dataset folders
        for odf_entry in oracle_dataset_folders:
            # Create full path
            odf_full_Path = os.path.join(self.stats_folder, odf_entry)

            # Get the explainer folders for that oracle-dataset combo
            explainer_folders_list = os.listdir(odf_full_Path)

            # Iterate over all explainer folders
            for exf_entry in explainer_folders_list:
                # Get the explainer folders full path
                exf_full_path = os.path.join(odf_full_Path, exf_entry)

                # get all the results files for that explainer-oracle-dataset combo
                result_files = os.listdir(exf_full_path)

                # iterate over the result files
                for result_entry in result_files:
                    result_file_full_path = os.path.join(exf_full_path, result_entry)

                    result.append(result_file_full_path)
            
                    
        return result


    def compare_graphs(self, data_instance_1, data_instance_2):
        """Given to graph G1 and G2 returns six lists containing: nodes_intersection, edges_intersection,
        nodes_only_in_G1, edges_only_in_G1, nodes_only_in_G2, and edges_only_in_G2"""
        G1 = data_instance_1.graph
        G2 = data_instance_2.graph

        intersection = nx.intersection(G1, G2)
        # Nodes in both graphs
        nodes_intersection = list(intersection.nodes())
        # Edges in both graphs
        edges_intersection = list(intersection.edges())
        # Nodes only in G1
        nodes_only_in_G1 = list(G1.nodes() - G2.nodes())
        # Edges only in G1
        edges_only_in_G1 = list(G1.edges() - G2.edges())
        # Nodes only in G2
        nodes_only_in_G2 = list(G2.nodes() - G1.nodes())
        # Edges only in G2
        edges_only_in_G2 = list(G2.edges() - G1.edges())

        return (nodes_intersection, edges_intersection, nodes_only_in_G1, 
                edges_only_in_G1, nodes_only_in_G2, edges_only_in_G2)

    
    def get_counterfactual_actions(self, data_instance, cf_data_instance):
        # Getting the similarities and the differences between the two graphs
        n_inter, e_inter, n_ori, e_ori, n_cf, e_cf = self.compare_graphs(data_instance, cf_data_instance)

        result = {'Remove Nodes': n_ori, 'Remove Edges': e_ori, 'Add Nodes': n_cf, 'Add Edges': e_cf}

        return result


    def draw_graph(self, data_instance, layout='random'):
        G = data_instance.graph

        edge_colors = ['cyan' for u, v in G.edges()]
        node_colors = ['cyan' for node in G.nodes()]

        # Applying the right layout to the graph
        if layout == 'circular':
            nx.draw_circular(G, node_color=node_colors, edge_color=edge_colors, with_labels=True)
        elif layout == 'kamada-kawai':
            nx.draw_kamada_kawai(G, node_color=node_colors, edge_color=edge_colors, with_labels=True)
        elif layout == 'planar':
            nx.draw_planar(G, node_color=node_colors, edge_color=edge_colors, with_labels=True)
        elif layout == 'random':
            nx.draw_random(G, node_color=node_colors, edge_color=edge_colors, with_labels=True)
        elif layout == 'spectral':
            nx.draw_spectral(G, node_color=node_colors, edge_color=edge_colors, with_labels=True)
        elif layout == 'spring':
            nx.draw_spring(G, node_color=node_colors, edge_color=edge_colors, with_labels=True)
        elif layout == 'shell':
            nx.draw_shell(G, node_color=node_colors, edge_color=edge_colors, with_labels=True)
        else:
            raise ValueError('Invalid graph layout')

        plt.show()


    def draw_counterfactual_actions(self, data_instance, cf_data_instance, layout='random'):

        nodes_shared, edges_shared, nodes_deleted, edges_deleted, nodes_added, edges_added = self.compare_graphs(data_instance, cf_data_instance)

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

        # Applying the right layout to the graph
        if layout == 'circular':
            nx.draw_circular(G, node_color=node_colors, edge_color=edge_colors, with_labels=True)
        elif layout == 'kamada-kawai':
            nx.draw_kamada_kawai(G, node_color=node_colors, edge_color=edge_colors, with_labels=True)
        elif layout == 'planar':
            nx.draw_planar(G, node_color=node_colors, edge_color=edge_colors, with_labels=True)
        elif layout == 'random':
            nx.draw_random(G, node_color=node_colors, edge_color=edge_colors, with_labels=True)
        elif layout == 'spectral':
            nx.draw_spectral(G, node_color=node_colors, edge_color=edge_colors, with_labels=True)
        elif layout == 'spring':
            nx.draw_spring(G, node_color=node_colors, edge_color=edge_colors, with_labels=True)
        elif layout == 'shell':
            nx.draw_shell(G, node_color=node_colors, edge_color=edge_colors, with_labels=True)
        else:
            raise ValueError('Invalid graph layout')

        plt.show()


    def draw_graph_custom_position(self, data_instance, position, img_store_address=None):
        G = data_instance.graph

        edge_colors = ['cyan' for u, v in G.edges()]
        node_colors = ['cyan' for node in G.nodes()]

        nx.draw(G=G, pos=position, node_color=node_colors, edge_color=edge_colors, with_labels=True)

        if img_store_address:
            plt.savefig(img_store_address, format='svg')

        plt.show(block=False)


    def draw_counterfactual_actions_custom_position(self, data_instance, cf_data_instance, position, img_store_address=None):
        nodes_shared, edges_shared, nodes_deleted, edges_deleted, nodes_added, edges_added = self.compare_graphs(data_instance, cf_data_instance)

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
        
        nx.draw(G=G, pos=position, node_color=node_colors, edge_color=edge_colors, with_labels=True)

        if img_store_address:
            plt.savefig(img_store_address, format='svg')

        plt.show(block=False)