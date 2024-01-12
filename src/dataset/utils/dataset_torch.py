import numpy as np

from typing import List

import torch
from torch_geometric.data import Data, Dataset

from src.dataset.instances.graph import GraphInstance
    
class TorchGeometricDataset(Dataset):
  
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