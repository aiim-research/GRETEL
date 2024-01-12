from numbers import Number
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, DenseGraphConv
from torch_geometric.data import Data


from src.dataset.instances.graph import GraphInstance
from src.core.explainer_base import Explainer
from src.core.trainable_base import Trainable
from src.dataset.utils.dataset_torch import TorchGeometricDataset
from src.utils.logger import GLogger
from src.utils.utils import pad_adj_matrix


class CLEARExplainer(Trainable, Explainer):

    def init(self):
        super().init()        
        self.batch_size = self.local_config['parameters']['batch_size']
        self.h_dim = self.local_config['parameters']['h_dim']
        self.z_dim = self.local_config['parameters']['z_dim']
        self.dropout = self.local_config['parameters']['dropout']
        self.encoder_type = self.local_config['parameters']['encoder_type']
        self.graph_pool_type = self.local_config['parameters']['graph_pool_type']
        self.disable_u = self.local_config['parameters']['disable_u']
        self.epochs = self.local_config['parameters']['epochs']
        self.alpha = self.local_config['parameters']['alpha']
        self.feature_dim = self.local_config['parameters']['feature_dim']
        self.lr = self.local_config['parameters']['lr']
        self.weight_decay = self.local_config['parameters']['weight_decay']
        self.lambda_sim = self.local_config['parameters']['lambda_sim']
        self.lambda_kl = self.local_config['parameters']['lambda_kl']
        self.lambda_cfe = self.local_config['parameters']['lambda_cfe']
        self.beta_x = self.local_config['parameters']['beta_x']
        self.beta_adj = self.local_config['parameters']['beta_adj']
        self.n_nodes = self.local_config['parameters']['n_nodes']

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = CLEAR(feature_dim=self.feature_dim,
                           graph_pool_type=self.graph_pool_type,
                           encoder_type=self.encoder_type,
                           n_nodes=self.n_nodes,
                           h_dim=self.h_dim,
                           z_dim=self.z_dim,
                           dropout=self.dropout,
                           disable_u=self.disable_u,
                           device=self.device).to(self.device)
                        
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)

        self._logger = GLogger.getLogger()
        
    def explain(self, instance):        
        self.model.eval()
        
        with torch.no_grad():
            # pad the adjacency matrix of the current instance
            padded_adj = pad_adj_matrix(instance.data, self.n_nodes)
            # create a new instance
            new_instance = GraphInstance(id=instance.id,
                                        label=instance.label,
                                        data=padded_adj,
                                        dataset=instance._dataset)
            # redo the manipulators
            instance._dataset.manipulate(new_instance)
            # get the features, adj matrix, graph causality and label
            features = torch.from_numpy(np.array(new_instance.node_features)).float().to(self.device)[None,:,:]
            adj = torch.from_numpy(new_instance.data).float().to(self.device)[None,:,:]
            u = torch.from_numpy(np.array(new_instance.graph_features[self.dataset.graph_features_map["graph_causality"]])).float().to(self.device)[None,:]
            labels = torch.from_numpy(np.array([new_instance.label])).to(self.device)[None,:]
            
            model_return = self.model(features, u, adj, labels)
            adj_reconst, features_reconst = model_return['adj_reconst'], model_return['features_reconst']
            
            adj_reconst_binary = torch.bernoulli(adj_reconst.squeeze())
            
            cf_instance = GraphInstance(id=instance.id,
                                        label=instance.label,
                                        data=adj_reconst_binary.to("cpu").detach().numpy(),
                                        node_features=features_reconst.squeeze().to("cpu").detach().numpy())
            
            return cf_instance

    def real_fit(self):
        train_loader = self.dataset.get_torch_loader(fold_id=self.fold_id,
                                                     batch_size=self.batch_size,
                                                     dataset_kls='src.explainer.generative.clear.CLEARGeometricDataset',
                                                     max_nodes=self.n_nodes)
        for epoch in range(self.epochs):
            self.model.train()
            
            batch_num = 0
            loss, loss_kl, loss_sim, loss_cfe, loss_kl_cf = 0, 0, 0, 0, 0
            for data in train_loader:
                batch_num += 1
                # here we're abusing the geometric dataset signature                                
                features = data.x.float().to(self.device)
                features = features.reshape(self.batch_size, -1, features.shape[-1])
                num_nodes = features.shape[1]
                u = data.edge_attr.float().to(self.device)[:,None]
                adj = data.edge_index.float().to(self.device)
                adj = adj.reshape(self.batch_size, num_nodes, num_nodes)
                labels = (1 - data.y.float()).to(self.device)[:,None]
                ########################################################
                self.optimizer.zero_grad()
                # forward pass
                retr = self.model(features, u, adj, labels)
                # z_cf
                z_mu_cf, z_logvar_cf = self.model.encoder(
                    retr['features_reconst'], 
                    u, 
                    retr['adj_reconst'], 
                    labels)
                # compute loss
                loss_params = {
                    'model': self.model,
                    'oracle': self.oracle,
                    'adj_input': adj,
                    'features_input': features,
                    'y_cf': labels,
                    'z_mu_cf': z_mu_cf,
                    'z_logvar_cf': z_logvar_cf
                }
                loss_params.update(retr)
                
                loss_results = self.__compute_loss(loss_params)
                loss_batch, loss_kl_batch, loss_sim_batch, loss_cfe_batch, loss_kl_batch_cf = loss_results['loss'],\
                    loss_results['loss_kl'], loss_results['loss_sim'], loss_results['loss_cfe'], loss_results['loss_kl_cf']
                    
                loss += loss_batch
                loss_kl += loss_kl_batch
                loss_sim += loss_sim_batch
                loss_cfe += loss_cfe_batch
                loss_kl_cf += loss_kl_batch_cf
                
            loss, loss_kl, loss_sim, loss_cfe, loss_kl_cf = loss / batch_num, loss_kl / batch_num, loss_sim / batch_num, loss_cfe / batch_num, loss_kl_cf / batch_num
            
            self.context.logger.info(f'Epoch {epoch+1} ---> loss {loss}')
            # backward
            alpha = self.alpha if epoch >= 450 else 0
            ((loss_sim + loss_kl + alpha * loss_cfe) / batch_num).backward()        
            self.optimizer.step()
        
        self.model._fitted = True
        
    def __compute_loss(self, params):
        _, oracle, z_mu, z_logvar, adj_permuted, features_permuted, adj_reconst, features_reconst, \
            _, _, y_cf, z_u_mu, z_u_logvar, z_mu_cf, z_logvar_cf = params['model'], params['oracle'], params['z_mu'], \
                params['z_logvar'], params['adj_permuted'], params['features_permuted'], params['adj_reconst'], params['features_reconst'], \
                    params['adj_input'], params['features_input'], params['y_cf'], params['z_u_mu'], params['z_u_logvar'], params['z_mu_cf'], params['z_logvar_cf']
                    
        # kl loss
        loss_kl = 0.5 * (((z_u_logvar - z_logvar) + ((z_logvar.exp() + (z_mu - z_u_mu).pow(2)) / z_u_logvar.exp())) - 1)
        loss_kl = torch.mean(loss_kl)
        
        # similarity loss
        size = len(features_permuted)
        dist_x = torch.mean(self.__distance_feature(features_permuted.view(size, -1), features_reconst.view(size, -1)))
        adj_permuted /= torch.max(adj_permuted)
        dist_a = self.__distance_graph_prob(adj_permuted, adj_reconst)
                
        loss_sim = self.beta_x * dist_x + self.beta_adj * dist_a
        
        # CFE loss
        y_pred = []
        for i in range(len(adj_reconst)):
            temp_instance = GraphInstance(id=-1,
                                          label=None,
                                          data=adj_reconst[i].to("cpu").detach().numpy().squeeze(),
                                          node_features=features_reconst[i].to("cpu").detach().numpy().squeeze())
            y_pred.append(np.array(oracle.predict_proba(temp_instance)))

        y_pred = torch.from_numpy(np.array(y_pred)).float().squeeze()
        loss_cfe = F.nll_loss(F.log_softmax(y_pred, dim=-1), y_cf.to("cpu").view(-1).long())
        
        # rep loss
        if z_mu_cf is None:
            loss_kl_cf = 0
        else:
            loss_kl_cf = 0.5 * (((z_logvar_cf - z_logvar) + ((z_logvar.exp() + (z_mu - z_mu_cf).pow(2)) / z_logvar_cf.exp())) - 1)
            loss_kl_cf = torch.mean(loss_kl_cf)
            
        loss = self.lambda_sim * loss_sim + self.lambda_kl * loss_kl + self.lambda_cfe * loss_cfe

        loss_results = {'loss': loss, 'loss_kl': loss_kl, 'loss_sim': loss_sim, 'loss_cfe': loss_cfe, 'loss_kl_cf':loss_kl_cf}
        return loss_results  
    
    
    def __distance_feature(self, feat_1, feat_2):
        pdist = nn.PairwiseDistance(p=2)
        return pdist(feat_1, feat_2) / 4
    
    def __distance_graph_prob(self, adj_1, adj_2_prob):
        return F.binary_cross_entropy(adj_2_prob, adj_1)
    
    
    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['batch_size'] =  self.local_config['parameters'].get('batch_size', 8)
        self.local_config['parameters']['h_dim'] =  self.local_config['parameters'].get('h_dim', 10)
        self.local_config['parameters']['z_dim'] =  self.local_config['parameters'].get('z_dim', 10)
        self.local_config['parameters']['dropout'] =  self.local_config['parameters'].get('dropout', .1)
        self.local_config['parameters']['encoder_type'] =  self.local_config['parameters'].get('encoder_type', 'gcn')
        self.local_config['parameters']['graph_pool_type'] =  self.local_config['parameters'].get('graph_pool_type', 'mean')
        self.local_config['parameters']['disable_u'] =  self.local_config['parameters'].get('disable_u', False)
        self.local_config['parameters']['epochs'] =  self.local_config['parameters'].get('epochs', 200)
        self.local_config['parameters']['alpha'] =  self.local_config['parameters'].get('alpha', 5)
        self.local_config['parameters']['lr'] =  self.local_config['parameters'].get('lr', 1e-3)
        self.local_config['parameters']['weight_decay'] =  self.local_config['parameters'].get('weight_decay', 1e-5)
        self.local_config['parameters']['lambda_sim'] =  self.local_config['parameters'].get('lambda_sim', 1)
        self.local_config['parameters']['lambda_kl'] =  self.local_config['parameters'].get('lambda_kl', 1)
        self.local_config['parameters']['lambda_cfe'] =  self.local_config['parameters'].get('lambda_cfe', 1)
        self.local_config['parameters']['beta_x'] =  self.local_config['parameters'].get('beta_x', 10)
        self.local_config['parameters']['beta_adj'] =  self.local_config['parameters'].get('beta_adj', 10)

        n_nodes = max([x.num_nodes for x in self.dataset.instances])
        self.local_config['parameters']['n_nodes'] = n_nodes

        self.local_config['parameters']['feature_dim'] = len(self.dataset.node_features_map)
    
class CLEAR(nn.Module):

    def __init__(self,
                 feature_dim,
                 h_dim=16,
                 z_dim=16,
                 dropout=False,
                 n_nodes=10,
                 encoder_type='gcn',
                 graph_pool_type='mean',
                 disable_u=False,
                 device='cuda'
                ):
        super(CLEAR, self).__init__()
        
        self.x_dim = feature_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.dropout = dropout
        self.n_nodes = n_nodes
        self.encoder_type = encoder_type
        self.graph_pool_type = graph_pool_type
        self.disable_u = disable_u
        self.device = device
        self.u_dim = 1 # init_params['u_dim']
        
        if self.disable_u:
            self.u_dim = 0
        if self.encoder_type == 'gcn':
            self.graph_model = DenseGCNConv(self.x_dim, self.h_dim)
        else:
            self.graph_model = DenseGraphConv(self.x_dim, self.h_dim)
        
        
        # prior
        self.prior_mean = MLP(self.u_dim, self.z_dim, self.h_dim,
                              n_layers=1, activation='none', slope=.1,
                              device=self.device)
        
        self.prior_var = nn.Sequential(
            MLP(self.u_dim, self.z_dim, self.h_dim, n_layers=1,
                activation='none', slope=.1, device=self.device),
            nn.Sigmoid()
        )
        
        # encoder
        self.encoder_mean = nn.Sequential(
            nn.Linear(self.h_dim + self.u_dim + 1, self.z_dim),
            nn.BatchNorm1d(self.z_dim),
            nn.ReLU()
        )
        
        
        self.encoder_var = nn.Sequential(
            nn.Linear(self.h_dim + self.u_dim + 1, self.z_dim),
            nn.BatchNorm1d(self.z_dim),
            nn.ReLU(),
            nn.Sigmoid()
        )
                
        # decoder
        self.decoder_x = nn.Sequential(
            nn.Linear(self.z_dim + 1, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.n_nodes * self.x_dim)
        )
        
        in_channels_a = self.z_dim + 1 if self.disable_u else self.z_dim + 2
        self.decoder_a = nn.Sequential(
            nn.Linear(in_channels_a, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.n_nodes * self.n_nodes),
            nn.Sigmoid()
        )
        self.graph_norm = nn.BatchNorm1d(self.h_dim)
        
        
    def encoder(self, features, u, adj, y_cf):
        # Q(Z | X, U, A, Y^CF)
        # input: x, u, A, y^cf
        # output: z
        graph_rep = self.graph_model(features, adj) # n x num_node x h_dim
        graph_rep  = self.graph_pooling(graph_rep, self.graph_pool_type) # n x h_dim
        graph_rep = self.graph_norm(graph_rep)
        
        if self.disable_u:
            z_mu = self.encoder_mean(torch.cat((graph_rep, y_cf), dim=1))
            z_logvar = self.encoder_var(torch.cat((graph_rep, y_cf), dim=1))
        else:
            z_mu = self.encoder_mean(torch.cat((graph_rep, u, y_cf), dim=1))
            z_logvar = self.encoder_var(torch.cat((graph_rep, u, y_cf), dim=1))
            
        return z_mu, z_logvar
    
    def decoder(self, z, y_cf, u):
        if self.disable_u:
            adj_reconst = self.decoder_a(
                torch.cat((z, y_cf), dim=1)
                ).view(-1, self.n_nodes, self.n_nodes)
        else:
            adj_reconst = self.decoder_a(
                torch.cat((z, u, y_cf), dim=1)
                ).view(-1, self.n_nodes, self.n_nodes)
                            
        features_reconst = self.decoder_x(
            torch.cat((z, y_cf), dim=1)
            ).view(-1, self.n_nodes, self.x_dim)
        
        return features_reconst, adj_reconst
    
    def graph_pooling(self, x, type='mean'):
        if type == 'max':
            out, _ = torch.max(x, dim=1, keepdim=False)
        elif type == 'sum':
            out = torch.sum(x, dim=1, keepdim=False)
        elif type == 'mean':
            out = torch.sum(x, dim=1, keepdim=False)
        return out
    
    def prior_params(self, u): # P(Z | U)
        if self.disable_u:
            z_u_mu = torch.zeros((len(u), self.h_dim)).to(self.device)
            z_u_logvar = torch.ones((len(u), self.h_dim)).to(self.device)
        else:
            z_u_logvar = self.prior_var(u)
            z_u_mu = self.prior_mean(u)
            
        return z_u_mu, z_u_logvar
    
    
    def reparametrize(self, mu, logvar):
        # compute z = mu + std * epsilon
        if self.training:
            # compute std from logvar
            std = torch.exp(0.5 * logvar)
            # sample epsilon from a normal distribution with
            # mean 0 and variance 1
            eps = torch.rand_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
        
    def forward(self, features, u, adj, y_cf):
        u_onehot = u
        
        z_u_mu, z_u_logvar = self.prior_params(u_onehot)
        # encoder
        z_mu, z_logvar = self.encoder(features, u_onehot, adj, y_cf)
        # reparametrize
        z_sample = self.reparametrize(z_mu, z_logvar)
        # decoder
        
        features_reconst, adj_reconst = self.decoder(z_sample, y_cf, u_onehot)
        
        return {
            'z_mu': z_mu,
            'z_logvar': z_logvar,
            'adj_permuted': adj,
            'features_permuted': features,
            'adj_reconst': adj_reconst,
            'features_reconst': features_reconst,
            'z_u_mu': z_u_mu,
            'z_u_logvar': z_u_logvar
        }
        
        

class MLP(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers,
                 activation='none', slope=.1, device='cuda'):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.slope = slope
        self.device = device
        
        
        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError(f'Wrong argument type for hidden_dim {hidden_dim}')
        
        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.hidden_dim = activation
        else:
            raise ValueError(f'Wrong argument type for activation {activation}')
        
        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError(f'Incorrect activation: {act}')

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
                
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
            
        self.fc = nn.ModuleList(_fc_list)
        
        
    @staticmethod
    def xtanh(x, alpha=.1):
        return x.tanh() + alpha * x
    
    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h
    
    
class CLEARGeometricDataset(TorchGeometricDataset):
  
    def __init__(self, instances: List[GraphInstance], max_nodes=10):
        self.max_nodes = max_nodes
        super(CLEARGeometricDataset, self).__init__(instances)
      
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
    def to_geometric(self, instance: GraphInstance, label=0) -> Data:   
      adj = torch.from_numpy(instance.data).double()
      x = torch.from_numpy(instance.node_features).double()
      label = torch.tensor(label).long()
      causality = np.array(instance.graph_features[instance._dataset.graph_features_map["graph_causality"]])
      return Data(x=x, y=label,
                  edge_index=adj,
                  edge_attr=torch.from_numpy(causality))