import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from src.utils.cfg_utils import default_cfg
from src.utils.torch.graph_pooling import TopKPooling

class TopKPoolingDiscriminator(nn.Module):
    
    def __init__(self, num_nodes, node_features, dim=2):
        super(TopKPoolingDiscriminator, self).__init__()
        self.training = False
        
        self.conv = GCNConv(node_features, dim).double()
        self.pool = TopKPooling(in_channels=dim, k=num_nodes)
        self.fc = nn.Linear(num_nodes * dim, 1).double()
        
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def set_training(self, training):
        self.training = training
        
    def forward(self, x, edge_list, edge_attr):
        x = x.double()
        edge_attr = edge_attr.double()
        x = self.conv(x, edge_list, edge_attr)
        
        if self.training:
            x = self.add_gaussian_noise(x)
        x = F.relu(x)
        x, _, _, _, _, _ = self.pool(x, edge_list, edge_attr)
        x = torch.flatten(x)
        x = self.fc(x)
        x = torch.sigmoid(x)

        return x
    
    def add_gaussian_noise(self, x, sttdev=0.2):
        noise = torch.randn(x.size(), device=self.device).mul_(sttdev)
        return x + noise
        
    @default_cfg
    def grtl_default(kls, num_nodes, node_features, dim=2):
        return {"class": kls,
                        "parameters": {
                            "node_features": node_features,
                            "num_nodes": num_nodes,
                            "dim": dim
                        }
        }
        
class SimpleDiscriminator(nn.Module):
    
    def __init__(self, num_nodes, node_features, dim=2):
        """This class provides a GCN to discriminate between real and generated graph instances"""
        super(SimpleDiscriminator, self).__init__()

        self.training = False
        
        self.conv1 = GCNConv(node_features, dim).double()
        self.fc = nn.Linear(num_nodes * dim, 1).double()
        
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def set_training(self, training):
        self.training = training

    def forward(self, x, edge_list, edge_attr):
        x = x.double()
        edge_attr = edge_attr.double()
        x = self.conv1(x, edge_list, edge_attr)
        
        if self.training:
            x = self.add_gaussian_noise(x)

        x = F.relu(x)
        x = F.dropout(x, p=.4, training=self.training)
        x = torch.flatten(x)
        x = self.fc(x)
        x = torch.sigmoid(x).squeeze()

        return x
    
    def add_gaussian_noise(self, x, sttdev=0.2):
        noise = torch.randn(x.size(), device=self.device).mul_(sttdev)
        return x + noise
        
    @default_cfg
    def grtl_default(kls, num_nodes, node_features, dim=2):
        return {"class": kls,
                        "parameters": {
                            "num_nodes": num_nodes,
                            "node_features": node_features,
                            "dim": dim
                        }
        }
        