from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from src.utils.cfg_utils import default_cfg
from src.utils.torch.graph_pooling import TopKPooling

class GraphEmbedder(nn.Module):
    
    def __init__(self, num_nodes, node_feature_dim, dim=2):
        """This class provides a GCN to discriminate between real and generated graph instances"""
        super(GraphEmbedder, self).__init__()

        self.training = False
        
        self.conv = GCNConv(node_feature_dim, dim).double()
        self.pool = TopKPooling(in_channels=dim, k=num_nodes)
        
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
        x = F.relu(x)

        return x
    
    def add_gaussian_noise(self, x, sttdev=0.2):
        noise = torch.randn(x.size(), device=self.device).mul_(sttdev)
        return x + noise
        
    @default_cfg
    def grtl_default(kls, num_nodes, node_feature_dim, dim=2):
        return {"class": kls,
                        "parameters": {
                            "num_nodes": num_nodes,
                            "node_feature_dim": node_feature_dim,
                            "dim": dim
                        }
        }
        
class PreDiscriminatorEmbedder(nn.Module):

    def __init__(self, num_nodes, node_feature_dim, dim=2) -> None:
        # embeds the graph into node vectors
        self.embedder = GraphEmbedder(num_nodes, node_feature_dim, dim)
        # decodes the embedded node vectors into the original node feature space
        self.node_decoder = nn.Linear(dim, node_feature_dim)
        # decodes the edge embeddings (concatenation of two node vectors)
        self.edge_decoder = nn.Linear(2 * dim, 1)

    def forward(self, node_features, edge_list, edge_attrs) -> Tuple[torch.Tensor, 
                                                                     torch.Tensor,
                                                                     torch.Tensor,
                                                                     torch.Tensor]:
        emb_nodes: torch.Tensor = self.embedder(node_features, edge_list, edge_attrs)
        # repeat the node embeddings n times where n is the number of nodes 
        interleaved = torch.repeat_interleave(emb_nodes, repeats=emb_nodes.shape[0], dim=0)
        repeated = emb_nodes.repeat(emb_nodes.shape[0])
        # where all rows are zero, then there's a self-loop, which we need to delete
        loops = interleaved - repeated
        non_empty_mask = loops.abs().sum(dim=0).bool()
        # remove the self-loops
        interleaved, repeated = interleaved[:,non_empty_mask], repeated[:,non_empty_mask]
        # create edge embeddings
        edge_embeddings = torch.concat([interleaved, repeated], dim=1)
        # check if the edge exists
        edge_exists = torch.empty(size=(edge_embeddings.shape[0], 1))
        for i, edge_embedding in enumerate(edge_embeddings):
            edge_exists[i] = torch.sigmoid(self.edge_decoder(edge_embedding))
        # reconstruct the node features
        recons_nodes = self.node_decoder(emb_nodes)
        return emb_nodes, recons_nodes, edge_embeddings, edge_exists
    