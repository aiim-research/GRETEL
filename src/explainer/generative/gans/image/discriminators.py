import torch
import torch.nn as nn
from torch.nn import LeakyReLU

from src.utils.cfg_utils import default_cfg
from src.core.factory_base import build_w_params_string


class SimpleDiscriminator(nn.Module):
    
    def __init__(self, num_nodes, kernel=(3,3), stride=(2,2), activation=LeakyReLU(), dropout_p=0.2):
        """This class discriminates between real and generated graph instances"""
        super(SimpleDiscriminator, self).__init__()

        self.training = False
        
        self.num_nodes = num_nodes
        
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=kernel, stride=stride)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel, stride=stride)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel, stride=stride)
        
        self.act = build_w_params_string(activation)
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.flatten = nn.Flatten()

        self.fc_len = self.init_run()
        self.fc = nn.Linear(64 * self.fc_len**2, 1)
       
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def init_run(self):
        with torch.no_grad():
            dummy_input = torch.randn((1,1, self.num_nodes, self.num_nodes)).to(self.device)
            x = self.conv3(self.conv2(self.conv1(dummy_input)))
        return x.shape[-1]

    def add_gaussian_noise(self, x, sttdev=0.2):
        noise = torch.randn(x.size(), device=self.device).mul_(sttdev)
        return x + noise

    def get_same_padding(self, kernel_size, stride, input_dim):
        return 0 if stride != 1 else (kernel_size - stride) // 2
    
    def set_training(self, training):
        self.training = training

    def forward(self, graph):
        x = self.conv1(graph)
        if self.training:
            x = self.add_gaussian_noise(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.act(self.conv2(x))
        x = self.dropout(x)
        x = self.act(self.conv3(x))
        x = self.dropout(x)
        x = self.flatten(x)
        x = torch.sigmoid(self.fc(x))
        return x
    
        
    @default_cfg
    def grtl_default(kls, num_nodes):
        return {"class": kls,
                        "parameters": {
                            "num_nodes": num_nodes
                        }
        }
        