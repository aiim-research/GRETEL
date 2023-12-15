import torch.nn as nn
from torch_geometric.nn.aggr import MeanAggregation,SoftmaxAggregation
from torch_geometric.nn.conv import GCNConv

from src.core.factory_base import build_w_params_string

class GCN(nn.Module):
   
    def __init__(self, node_features, num_conv_layers=2, conv_booster=1, pooling=MeanAggregation):
        super(GCN, self).__init__()
        
        self.in_channels = node_features
        self.out_channels = int(self.in_channels * conv_booster)
          
        self.pooling =  build_w_params_string(pooling)
 
        
        if num_conv_layers>1:
            self.num_conv_layers = [(self.in_channels, self.out_channels)] + [(self.out_channels, self.out_channels) * (num_conv_layers - 1)]
        else:
            self.num_conv_layers = [(self.in_channels, self.out_channels)]
        self.graph_convs = self.__init__conv_layers()
        
    def forward(self, node_features, edge_index, edge_weight, batch):
        # convolution operations
        for conv_layer in self.graph_convs[:-1]:
            node_features = conv_layer(node_features, edge_index, edge_weight)
            node_features = nn.functional.relu(node_features)

        # global pooling
        if isinstance(self.graph_convs[-1],nn.Identity):
            return self.graph_convs[-1](node_features)
        return self.graph_convs[-1](node_features, batch)
    
    def __init__conv_layers(self):
        ############################################
        # initialize the convolutional layers interleaved with pooling layers
        graph_convs = []
        for i in range(len(self.num_conv_layers)):#add len
            graph_convs.append(GCNConv(in_channels=self.num_conv_layers[i][0],
                                      out_channels=self.num_conv_layers[i][1]).double())
        graph_convs.append(self.pooling)
        return nn.Sequential(*graph_convs).double()
