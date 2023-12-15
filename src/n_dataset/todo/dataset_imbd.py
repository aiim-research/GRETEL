from src.dataset.instances.base import DataInstance
from src.dataset.data_instance_features import DataInstanceWFeatures
from src.dataset.dataset_base import Dataset
from scipy.io import loadmat
import numpy as np

class IMDBDataset(Dataset):

    def __init__(self, id, self_loops=False, config_dict=None) -> None:
        
        super().__init__(id, config_dict)
        
        self.self_loops = self_loops
        
        self.instances = []
        self.name = 'imdb_dataset'

    def read_adjacency_matrices(self, dataset_path):
        """
        Reads the dataset from the adjacency matrices
        """
        data = loadmat(dataset_path)
        assert data != None
        self.preprocess(data)
            
            
    def preprocess(self, data):        
        adj_all_orin = data['graph_struct'].squeeze()
        labels = data['labels'].squeeze()
        
        for i in range(len(adj_all_orin)):
            adj = adj_all_orin[i][0]
            if self.self_loops:
                adj = adj + np.eye(adj.shape[0])
                
            inst = DataInstance(i)
            inst.from_numpy_array(adj)
            inst.graph_label = labels[i][0]
            
            self.instances.append(inst)
            
            
class IMDBClear(IMDBDataset):
    
    def __init__(self, id, max_nodes=15, feature_dim=8,
                 padding=False, self_loops=False,
                 config_dict=None) -> None:
        
        super().__init__(id, self_loops, config_dict)
        
        self.name = 'imdb_clear_dataset'
        
        self.max_nodes = max_nodes
        self.feature_dim = feature_dim
        self.padding = padding
        
    def preprocess(self, data):
        super().preprocess(data)
        # filter those instances that exceed the maximum number of nodes
        self.instances = list(filter(lambda i: i.graph.number_of_nodes() <= self.max_nodes), self.instances)
        # calculate the degree of each node per instance
        degrees = [i.node_degrees() for i in self.instances]
        avg_degree = np.mean(degrees)
        # get the maximum number of nodes present in the dataset
        max_node_number_in_dataset = 0
        # change the instances to DataInstanceWFeatures
        # where the feature vector is x \in R^{n_nodes, feature_dim}
        for i in range(len(self.instances)):
            curr_inst = self.instances[i]
            curr_inst_n_nodes = curr_inst.graph.number_of_nodes()
            graph_label = 1  if np.mean(degrees[i]) >= avg_degree else 0
            self.instances[i] = DataInstanceWFeatures(id=i,
                                                      graph=curr_inst.graph,
                                                      feaures=np.random.normal(0, 1, (curr_inst_n_nodes, self.feature_dim)),
                                                      graph_dgl=curr_inst.graph_dgl,
                                                      node_labels=curr_inst.node_labels,
                                                      edge_labels=curr_inst.edge_labels,
                                                      graph_label=graph_label
            )
            max_node_number_in_dataset = max(max_node_number_in_dataset, curr_inst_n_nodes)
        
        # padd the adjacency matrices and feature vectors
        if self.padding:
            for i in range(len(self.instances)):
                num_nodes = self.instances[i].graph.number_of_nodes()
                # pad adjacency matrix
                adj_matrix = self.instances[i].to_numpy_array()
                adj_padded = np.eye(max_node_number_in_dataset) if self.self_loops else np.zeros(max_node_number_in_dataset)
                adj_padded[:num_nodes, :num_nodes] = adj_matrix
                self.instances[i].from_numpy_array(adj_padded)
                # pad features
                features = self.instances[i].features
                features_padded = np.zeros((max_node_number_in_dataset, self.feature_dim))
                features_padded[:num_nodes] = features
                self.instances[i].features = features_padded
        