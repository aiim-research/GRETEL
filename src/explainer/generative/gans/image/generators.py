
import torch
import torch.nn as nn
from torch.nn import LeakyReLU

from src.core.factory_base import build_w_params_string
from src.utils.cfg_utils import default_cfg


class ResGenerator(nn.Module):
    
    def __init__(self, num_nodes,
                 conv_kernel=(3,3), conv_stride=(2,2),
                 deconv_kernel=(4,4), deconv_stride=(2,2),
                 activation=LeakyReLU(),
                 dropout_p=0.2,
                 residuals=True):
        
        super(ResGenerator, self).__init__()
        
        self.num_nodes = num_nodes
        
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=conv_kernel, stride=conv_stride)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=conv_kernel, stride=conv_stride)
        
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=deconv_kernel, stride=deconv_stride, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=deconv_kernel, stride=deconv_stride, padding=1)
        
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(7,7), stride=(1,1), padding=self.__get_same_padding(7,1))
        
        self.flatten = nn.Flatten()
        self.act =  build_w_params_string(activation)
        self.dropout = nn.Dropout2d(p=dropout_p)

        self.flatten = nn.Flatten()

        self.fc_len = self.init_run()
        
        self.fc = nn.Linear(in_features=64 * self.fc_len**2,
                            out_features=128 * (self.num_nodes // 4)**2)
        
        self.residuals = residuals
        self.training = False
        
        
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
        
    def set_training(self, training):
        self.training = training

    def init_run(self):
        with torch.no_grad():
            dummy_input = torch.randn((1,1, self.num_nodes, self.num_nodes)).to(self.device)
            x = self.conv2(self.conv1(dummy_input))
        return x.shape[-1]

    def forward(self, graph):
        x = self.act(self.conv1(graph))
        x = self.dropout(x)
        x = self.act(self.conv2(x))
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.act(self.fc(x))
        x = x.view((-1, 128, self.num_nodes//4, self.num_nodes//4))
        x = self.act(self.deconv1(x))
        x = self.act(self.deconv2(x))
        x = self.final_conv(x)
        x = torch.tanh(x) # Tanh values are in [-1, 1] so allow the residual to add or remove edges
        # building the counterfactual example from the union of the residuals and the original instance
        if self.residuals:
            x = torch.add(graph, x)
        # delete self loops
        for i in range(x.shape[0]):
            x[i][0].fill_diagonal_(0)
            
        mask = ~torch.eye(self.num_nodes, dtype=bool)
        mask = torch.stack([mask] * x.shape[0])[:,None,:,:]
        x[mask] = torch.sigmoid(x[mask])
        
        x = (torch.rand_like(x) < x).to(torch.float)
                      
        return x


    def __get_same_padding(self, kernel_size, stride):
        return 0 if stride != 1 else (kernel_size - stride) // 2
    
    
    @default_cfg
    def grtl_default(kls, num_nodes):
        return {"class": kls, "parameters": { "num_nodes": num_nodes } }
    