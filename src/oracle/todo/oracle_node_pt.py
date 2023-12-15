import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.dataset.data_instance_node import NodeDataInstance
from src.dataset.dataset_node import NodeDataset
from src.core.oracle_base import Oracle
from src.utils import accuracy, normalize_adj
from torch.nn.parameter import Parameter
from torch.nn.utils import clip_grad_norm
from torch_geometric.utils import dense_to_sparse


class NodeOracle(Oracle):
	def __init__(self, id, oracle_store_path, config_dict=None, n_node_types=118) -> None:
		super().__init__(id, oracle_store_path, config_dict)
		self.name = 'node_oracle'
		self._clf = None
		self._n_node_types = n_node_types
		self._max_n_nodes = 0

	def fit(self, dataset: NodeDataset, split_i=0):
		pass
	
	def create_model(self):
		"""
		torch model requires data at initialization stage hence the model is created in the fit function.
		"""
		pass

	def _real_predict(self, data_instance: NodeDataInstance):
		self._call_counter -= 1
		return self.predict_all_nodes(data_instance)[data_instance.target_node].item()

	def predict_all_nodes(self, data_instance: NodeDataInstance):
		self._call_counter += 1
		data = data_instance.graph_data
		adj = torch.Tensor(data["adj"]).squeeze()
		features = torch.Tensor(data["feat"]).squeeze()
		labels = torch.tensor(data["labels"]).squeeze()
		idx_train = torch.tensor(data["train_idx"])
		idx_test = torch.tensor(data["test_idx"])
		norm_adj = normalize_adj(adj)

		model = self._clf
		model.eval()
		output = model(features, norm_adj)
		return torch.argmax(output, dim=1)


	def embedd(self, instance):
		return instance

	def get_torch_model(self):
		return self._clf

	def write_oracle(self):
		# Creating the rute to store the oracle
		oracle_uri = os.path.join(self._oracle_store_path, self.name, 'oracle.pt')
		torch.save(self._clf, oracle_uri)

	def read_oracle(self, oracle_name):
		# Creating the rute to stored oracle
		oracle_uri = os.path.join(self._oracle_store_path, oracle_name, 'oracle.pt')
		self._clf = torch.load(oracle_uri)

	def eval(self):
		return self._clf.eval()

	def state_dict(self):
		return self._clf.state_dict().copy()

	def named_parameters(self):
		return self._clf.named_parameters()
