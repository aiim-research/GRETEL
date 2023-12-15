import os

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import from_networkx, mean_nodes
from dgl.nn.pytorch import GraphConv
from dgl.data import DGLDataset
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
from src.dataset.dataset_base import Dataset
from src.core.oracle_base import Oracle


class CF2Oracle(Oracle):
    
    def __init__(self, id, oracle_store_path,
                 converter,
                 in_dim=8, h_dim=6, 
                 lr=1e-4, weight_decay=0,
                 epochs=100,
                 fold_id=0,
                 threshold=.5,
                 batch_size_ratio=.1,
                 config_dict=None) -> None:
        
        super(CF2Oracle, self).__init__(id, oracle_store_path, config_dict)
        self._name = 'cf2'
        
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.epochs = epochs
        self.threshold = threshold
        self.batch_size_ratio = batch_size_ratio
        
        self.clf = ClfGCNGraph(in_dim=self.in_dim,
                               h_dim=self.h_dim)
        
        self.optimizer = torch.optim.Adam(self.clf.parameters(),
                                          lr=lr, weight_decay=weight_decay)
        
        self.loss_fn = nn.BCELoss()
        self.converter = converter
        self.fold_id = fold_id
                
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        

    def fit(self, dataset: Dataset, split_i=0):
        self._name = f'{self._name}_fit_on_{dataset.name}_fold_id={self.fold_id}'
        dataset = self.converter.convert(dataset)
        self.batch_size = int(len(dataset.instances) * self.batch_size_ratio)
        # If there is an available oracle trained on that dataset load it
        if os.path.exists(os.path.join(self._oracle_store_path, self._name)):
            self.read_oracle()
        else: # If not then train
            loader = self.__transform_data(dataset)
            
            for epoch in range(self.epochs):
                losses = []
                for batched_graph, labels in loader:
                    batched_graph = batched_graph.to(self.device)
                    labels = labels.to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    pred = self.clf(batched_graph, batched_graph.ndata['feat'].float(),
                                    batched_graph.edata['weights']).squeeze()
                    
                    loss = self.loss_fn(pred, labels.squeeze())
                    losses.append(loss.to('cpu').detach().numpy())
                    loss.backward()
                    
                    self.optimizer.step()
                
                print('epoch:%d' % epoch, 'loss:',  np.mean(losses))
            # Writing to disk the trained oracle
            self.write_oracle()
            
    def _real_predict(self, data_instance):
        return self._apply_threshold(self._real_predict_proba(data_instance))
    
    def _apply_threshold(self, num):
        return 1 if num >= self.threshold else 0
    
    def _real_predict_proba(self, data_instance):
        if not hasattr(data_instance, 'features')\
            or not hasattr(data_instance, 'weights'):
            data_instance = self.converter.convert_instance(data_instance)
            
        graph = data_instance.to_numpy_array()
        features = data_instance.features
        weights = data_instance.weights

        graph = from_networkx(nx.from_numpy_array(graph)).to(self.device)
        features = torch.from_numpy(features).float().to(self.device)
        weights = torch.from_numpy(weights).float().to(self.device)
        
        return self.clf(graph, features, weights).squeeze()    
    
    def embedd(self, instance):
        return instance
    
    def write_oracle(self):
        directory = os.path.join(self._oracle_store_path, self._name)
        if not os.path.exists(directory):
            os.mkdir(directory)
        torch.save(self.clf.state_dict(), os.path.join(directory, 'oracle'))
      
    def read_oracle(self):
        self.clf.load_state_dict(torch.load(
            os.path.join(self._oracle_store_path, self._name, 'oracle')))
        
    def __transform_data(self, dataset: Dataset):             
        adj  = np.array([i.to_numpy_array() for i in dataset.instances])
        features = np.array([i.features for i in dataset.instances])
        weights = np.array([i.weights for i in dataset.instances])
        y = np.array([i.graph_label for i in dataset.instances])
        
        indices = dataset.get_split_indices()[self.fold_id]['train'] 
              
        dgl_dataset = CustomDGLDataset(adj, features, weights, y)
                
        dataloader = GraphDataLoader(dgl_dataset, batch_size=self.batch_size, drop_last=False)

        return dataloader
        

class ClfGCNGraph(nn.Module):
   
    def __init__(self, in_dim, h_dim):
        super(ClfGCNGraph, self).__init__()
                
        self.conv1 = GraphConv(in_dim, h_dim, weight=False, allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_dim, h_dim, weight=False, allow_zero_in_degree=True)
        self.conv3 = GraphConv(h_dim, h_dim, weight=False, allow_zero_in_degree=True)
        self.dense1 = nn.Linear(h_dim, 16)
        self.dense2 = nn.Linear(16, 8)
        self.dense3 = nn.Linear(8, 1)
        
    def forward(self, graph, features, e_weight):
        h = F.relu(self.conv1(graph, features, edge_weight=e_weight))
        h = F.relu(self.conv2(graph, features, edge_weight=e_weight))
        h = F.relu(self.conv3(graph, features, edge_weight=e_weight))
        graph.ndata['h'] = h
        h = mean_nodes(graph, 'h')
        h = F.relu(self.dense1(h))
        h = F.relu(self.dense2(h))
        h = torch.sigmoid(self.dense3(h))
        return h
    
    
class CustomDGLDataset(DGLDataset):
    
    def __init__(self, adj_matrices, features, edge_weights, labels):
        super(CustomDGLDataset, self).__init__(name = 'custom_dgl_dataset')
        
        self.adj_matrices = adj_matrices
        self.features = features
        self.edge_weights = edge_weights
        self.labels = torch.from_numpy(labels).float()
        
        self.graphs = [None] * len(adj_matrices)
        
        
        for i in range(len(self.adj_matrices)):
            # create the DGL graph object from the adj matrix
            g = from_networkx(nx.from_numpy_array(self.adj_matrices[i]))
            # set node features of the graph
            g.ndata['feat'] = torch.from_numpy(self.features[i]).float()
            # set the edge weights of the graph
            g.edata['weights'] = torch.from_numpy(self.edge_weights[i]).float()
            # save the graph object
            self.graphs[i] = g
                    
    def process(self):
        print("Processing")
        
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.graphs)