import numpy as np

from typing import List

import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset as GeometricDataset
from torch.utils.data import Dataset

from src.dataset.instances.graph import GraphInstance
from src.utils.utils import pad_adj_matrix
    
class TorchGeometricDataset(GeometricDataset):
  
  def __init__(self, instances: List[GraphInstance]):
    super(TorchGeometricDataset, self).__init__()    
    self.instances = []
    self._process(instances)
    
  def len(self):
    return len(self.instances)
  
  def get(self, idx):
    return self.instances[idx]
  
  def _process(self, instances: List[GraphInstance]):
    self.instances = [self.to_geometric(inst, label=inst.label) for inst in instances]
      
  @classmethod
  def to_geometric(self, instance: GraphInstance, label=0) -> Data:   
    adj = torch.from_numpy(instance.data).double()
    x = torch.from_numpy(instance.node_features).double()
    a = torch.nonzero(adj).int()
    w = torch.from_numpy(instance.edge_weights).double()
    label = torch.tensor(label).long()
    return Data(x=x, y=label, edge_index=a.T, edge_attr=w)
  
  
class TorchDataset(Dataset):
  
    def __init__(self, instances: List[GraphInstance], max_nodes=10):
        super(TorchDataset, self).__init__()
        self.instances = instances
        self.max_nodes = max_nodes
        self._process(instances)
      
    def _process(self, instances: List[GraphInstance]):
        for i, instance in enumerate(instances):
            padded_adj = pad_adj_matrix(instance.data, self.max_nodes)
            # create a new instance
            new_instance = GraphInstance(id=instance.id,
                                        label=instance.label,
                                        data=padded_adj,
                                        dataset=instance._dataset)
            # redo the manipulators
            instance._dataset.manipulate(new_instance)
            instances[i] = new_instance
        
        self.instances = [self.to_geometric(inst, label=inst.label) for inst in instances]
 
    @classmethod
    def to_geometric(self, instance: GraphInstance, label=0):   
        adj = torch.from_numpy(instance.data).double()
        x = torch.from_numpy(instance.node_features).double()
        label = torch.tensor(label).long()
        return adj, x, label
    
    def __getitem__(self, index):
        return self.instances(index)