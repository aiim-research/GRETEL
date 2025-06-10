import torch
import torch.nn as nn

from src.explainer.generative.gans.graph.learnable_edges.graph_embedders import GraphEmbedder
from src.utils.cfg_utils import default_cfg

class TranslatingGenerator(nn.Module):

    def __init__(self, k: int, node_features: int, 
                 in_embed_dim: int=2, 
                 out_embed_dim: int=2, 
                 num_translator_layers: int=2,
                 gaussian_std: float=.1) -> None:
        super(TranslatingGenerator, self).__init__()
        self.embedder = GraphEmbedder(k, node_features, in_embed_dim)
        
        self.translator = nn.Sequential()
        emb_dim = in_embed_dim
        for _ in range(num_translator_layers):
            self.translator.append(nn.Linear(emb_dim, out_embed_dim))
            self.translator.append(nn.ReLU())
            emb_dim = out_embed_dim

        self.training = True

        self.gaussian_std = gaussian_std
        
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

    def forward(self, node_features, edge_list, edge_weights, batch=None):
        cf_node_embeddings = self.embedder(node_features, edge_list, edge_weights, batch)
        f_node_embeddings = self.translator(cf_node_embeddings)
        f_node_embeddings = self.add_gaussian_noise(f_node_embeddings, self.gaussian_std)
        return f_node_embeddings
    
    def set_training(self, training):
        self.training = training
    
    def add_gaussian_noise(self, x, sttdev=0.1):
        noise = torch.randn(x.size(), device=self.device).mul_(sttdev)
        return x + noise

    @default_cfg
    def grtl_default(kls, k: int,
                     node_features: int, 
                     in_embed_dim: int=2, 
                     out_embed_dim: int=2,
                     num_translator_layers: int=2):
        
        return {
            "class": kls,
            "parameters": {
                "k": k,
                "node_features": node_features,
                "in_embed_dim": in_embed_dim,
                "out_embed_dim": out_embed_dim,
                "num_translator_layers": num_translator_layers
            }
        }
