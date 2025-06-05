from src.utils.cfg_utils import default_cfg
from src.utils.torch.utils import rebuild_adj_matrix
import torch
import torch.nn as nn
from torch_geometric.nn import GAE

from src.utils.torch.gcn import GCN


class ResGenerator(nn.Module):
    
    def __init__(self, node_features, num_conv_layers=2, conv_booster=1, residuals=True):
        super(ResGenerator, self).__init__()
        
        self.node_features = node_features
        self.num_conv_layers = num_conv_layers
        self.conv_booster = conv_booster #conv_booster
        # encoder with no pooling
        self.encoder = GCN(self.node_features, self.num_conv_layers, self.conv_booster, nn.Identity()).double()
        self.feature_decoder = GCN(self.node_features * self.conv_booster, self.num_conv_layers, self.conv_booster**-1, nn.Identity()).double()
        # graph autoencoder with inner product decoder
        self.model = GAE(encoder=self.encoder).double()
        self.residuals = residuals
        
        self.device = (
            "cpu"
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
        self.encoder.training = training

    def forward(self, node_features, edge_list, edge_attr, batch):
        encoded_node_features = self.model.encode(node_features, edge_list, edge_attr, batch)
        edge_weights = self.model.decoder.forward_all(encoded_node_features, sigmoid=False)
        edge_weights = torch.nan_to_num(edge_weights, 0)
        
        decoded_node_features = self.feature_decoder(node_features, edge_list, edge_attr, batch)

        if self.residuals: #TODO: Fix the booster expansion if make sense
            #decoded_node_features = encoded_node_features.repeat(1, node_features.size(1) // encoded_node_features.size(1))
            decoded_node_features = torch.add(decoded_node_features, node_features)
            edge_weights = torch.add(edge_weights, rebuild_adj_matrix(len(node_features), edge_list, edge_attr.T, self.device))
            
        return decoded_node_features, edge_list, torch.sigmoid(edge_weights)
    
    @default_cfg
    def grtl_default(kls, node_features):
        return {"class": kls, "parameters": { "node_features": node_features } }