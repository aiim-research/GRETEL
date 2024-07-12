import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.cfg_utils import default_cfg

class EmbeddingDiscriminator(nn.Module):

    def __init__(self, num_nodes, dim=2, dropout=.4):
        """This class discriminates on the node embeddings"""
        super(EmbeddingDiscriminator, self).__init__()

        self.dropout = dropout
        self.training = False
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

    def forward(self, x):
        if self.training:
            x = self.add_gaussian_noise(x)

        x = torch.flatten(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        x = torch.sigmoid(x).squeeze()

        return x
    
    def add_gaussian_noise(self, x, sttdev=0.2):
        noise = torch.randn(x.size(), device=self.device).mul_(sttdev)
        return x + noise
        
    @default_cfg
    def grtl_default(kls, num_nodes, dim=2, dropout=.4):
        return {"class": kls,
                        "parameters": {
                            "num_nodes": num_nodes,
                            "dropout": dropout,
                            "dim": dim
                        }
        }
        
