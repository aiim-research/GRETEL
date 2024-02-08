from os.path import join
import numpy as np
import networkx as nx
from src.dataset.instances.graph import GraphInstance
from src.dataset.generators.base import Generator

class GithubStargazersGenerator(Generator):
    
    def init(self):
        self.data_path = self.local_config['parameters']['data_path']
        self.max_number_nodes = self.local_config['parameters']['max_number_nodes']
        self.generate_dataset()

    def check_configuration(self):
        super().check_configuration()
        local_config = self.local_config

        # set defaults
        local_config['parameters']['data_path'] = local_config['parameters'].get('data_path','data/datasets/github_stargazers')
        local_config['parameters']['max_number_nodes'] = local_config['parameters'].get('max_number_nodes', 200)

    def generate_dataset(self):
        if not len(self.dataset.instances):
            adj_matrix_path = join(self.data_path, 'github_stargazers_A.txt')
            graph_indicator_path = join(self.data_path, 'github_stargazers_graph_indicator.txt')
            graph_labels_path = join(self.data_path, 'github_stargazers_graph_labels.txt')

            edges = np.loadtxt(adj_matrix_path, delimiter=',',dtype=int)
            graph_indicator = np.loadtxt(graph_indicator_path, dtype=int)
            graph_labels = np.loadtxt(graph_labels_path, dtype=int)

            for graph_id in range(1, len(graph_labels) + 1):
                print(f"Generating graph {graph_id}")
                
                # Filter for the edges of the currnent graph
                graph_nodes = np.where(graph_indicator == graph_id)[0] + 1 # Add one to make up for the 0th index
                if len(graph_nodes) > self.max_number_nodes:
                    print(f"Graph {graph_id} skipped due to large size")
                    continue # skiping the biggest graphs to optimize needed resoureces

                graph_edges = edges[np.isin(edges, graph_nodes).any(axis=1)]

                G = nx.Graph()
                G.add_edges_from(graph_edges)
                graph = nx.to_numpy_array(G)

                label = graph_labels[graph_id - 1]
                self.dataset.instances.append(GraphInstance(id=graph_id, data=graph, label=label))    

    def get_num_instances(self):
        return len(self.dataset.instances)



