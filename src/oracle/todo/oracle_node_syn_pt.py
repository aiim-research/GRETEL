import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.oracle.todo.oracle_node_pt import NodeOracle
from src.dataset.data_instance_node import NodeDataInstance
from src.dataset.dataset_node import NodeDataset
from src.core.oracle_base import Oracle
from src.utils import accuracy, normalize_adj
from torch.nn.parameter import Parameter
from torch.nn.utils import clip_grad_norm
from torch_geometric.utils import dense_to_sparse


class GraphConvolution(nn.Module):
	"""
	Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
	"""

	def __init__(self, in_features, out_features, bias=True):
		super(GraphConvolution, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.FloatTensor(in_features, out_features))
		if bias:
			self.bias = Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input, adj):
		support = torch.mm(input, self.weight)
		output = torch.spmm(adj, support)
		if self.bias is not None:
			return output + self.bias
		else:
			return output

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
			   + str(self.in_features) + ' -> ' \
			   + str(self.out_features) + ')'


class GCNSynthetic(nn.Module):
	"""
	3-layer GCN used in GNN Explainer synthetic tasks
	"""
	def __init__(self, nfeat, nhid, nout, nclass, dropout):
		super(GCNSynthetic, self).__init__()

		self.gc1 = GraphConvolution(nfeat, nhid)
		self.gc2 = GraphConvolution(nhid, nhid)
		self.gc3 = GraphConvolution(nhid, nout)
		self.lin = nn.Linear(nhid + nhid + nout, nclass)
		self.dropout = dropout

	def forward(self, x, adj):
		x1 = F.relu(self.gc1(x, adj))
		x1 = F.dropout(x1, self.dropout, training=self.training)
		x2 = F.relu(self.gc2(x1, adj))
		x2 = F.dropout(x2, self.dropout, training=self.training)
		x3 = self.gc3(x2, adj)
		x = self.lin(torch.cat((x1, x2, x3), dim=1))
		return F.log_softmax(x, dim=1)

	def loss(self, pred, label):
		return F.nll_loss(pred, label)


class SynNodeOracle(NodeOracle):
	def __init__(self, id, oracle_store_path, config_dict=None, n_node_types=118) -> None:
		super().__init__(id, oracle_store_path, config_dict)
		self.name = 'gcn_syn_pt'
		self._clf = None
		self._n_node_types = n_node_types
		self._max_n_nodes = 0

		enable_config = config_dict is not None
		# set parameters for the torch model with defaults
		self._hidden = config_dict["hidden"] if enable_config and "hidden" in config_dict.keys() else 20 
		self._n_layers = config_dict["n_layers"] if enable_config and "n_layers" in config_dict.keys() else 3
		self._dropout = config_dict["dropout"] if enable_config and "hiddropoutden" in config_dict.keys() else 0.0
		self._seed = config_dict["seed"] if enable_config and "seed" in config_dict.keys() else 42
		self._lr = config_dict["lr"] if enable_config and "lr" in config_dict.keys() else 0.005
		self._optimizer = config_dict["optimizer"] if enable_config and "optimizer" in config_dict.keys() else "SGD"
		self._n_momentum = config_dict["n_momentum"] if enable_config and "n_momentum" in config_dict.keys() else 0.0
		self._beta = config_dict["beta"] if enable_config and "beta" in config_dict.keys() else 0.5
		self._num_epochs = config_dict["num_epochs"] if enable_config and "num_epochs" in config_dict.keys() else 500
		self._device = config_dict["device"] if enable_config and "device" in config_dict.keys() else 'cpu'
		self._clip = config_dict["clip"] if enable_config and "clip" in config_dict.keys() else 2.0
		self._weight_decay = config_dict["weight_decay"] if enable_config and "weight_decay" in config_dict.keys() else 0.001

	
	def create_model(self):
		"""
		torch model requires data at initialization stage hence the model is created in the fit function.
		"""
		pass

	def fit(self, dataset: NodeDataset, split_i=0):
		data = dataset.get_data()[0].graph_data

		# Creating the name of the folder for storing the trained oracle
		oracle_name = self.name + '_fit_on_' + dataset.name

		# Creating the rute to store the oracle
		oracle_uri = os.path.join(self._oracle_store_path, oracle_name)

		# Creating the name of the folder for storing the trained oracle
		oracle_name = self.name + '_fit_on_' + dataset.name

		# Creating the rute to store the oracle
		oracle_uri = os.path.join(self._oracle_store_path, oracle_name)

		if os.path.exists(oracle_uri):
			# Load the weights of the trained model
			self.name = oracle_name
			self.read_oracle(oracle_name)
		else:
			# Create the folder to store the oracle if it does not exist
			os.mkdir(oracle_uri)        
			self.name = oracle_name

			adj = torch.Tensor(data["adj"]).squeeze()       # Does not include self loops
			features = torch.Tensor(data["feat"]).squeeze()
			labels = torch.tensor(data["labels"]).squeeze()
			idx_train = torch.tensor(data["train_idx"])
			idx_test = torch.tensor(data["test_idx"])
			edge_index = dense_to_sparse(adj) 
			
			norm_adj = normalize_adj(adj)       # According to reparam trick from GCN paper

			model = GCNSynthetic(nfeat=features.shape[1], nhid=self._hidden, nout=self._hidden,
						nclass=len(labels.unique()), dropout=self._dropout)
			optimizer = optim.SGD(model.parameters(), lr=self._lr, weight_decay=self._weight_decay)

			if (dataset.name in ["syn1","syn4","syn5"]):
				model.load_state_dict(torch.load(F"/home/coder/gretel/data/pretrained/models/gcn_3layer_{dataset.name}.pt"))
			else:
				for epoch in range(self._num_epochs):
					model.train()
					optimizer.zero_grad()
					output = model(features, norm_adj)
					loss_train = model.loss(output[idx_train], labels[idx_train])
					y_pred = torch.argmax(output, dim=1)
					acc_train = accuracy(y_pred[idx_train], labels[idx_train])
					loss_train.backward()
					clip_grad_norm(model.parameters(), self._clip)
					optimizer.step()

			self._clf = model   
			self.write_oracle()
