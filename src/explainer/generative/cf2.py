import math
import numpy as np
import torch
from copy import deepcopy
from src.dataset.instances.graph import GraphInstance
from src.core.explainer_base import Explainer
from src.core.trainable_base import Trainable
from src.core.oracle_base import Oracle


class CF2Explainer(Trainable, Explainer):

    def init(self):
        self.n_nodes = self.local_config['parameters']['n_nodes']
        self.batch_size_ratio = self.local_config['parameters']['batch_size_ratio']
        self.lr = self.local_config['parameters']['lr']
        self.weight_decay = self.local_config['parameters']['weight_decay']
        self.gamma = self.local_config['parameters']['gamma']
        self.lam = self.local_config['parameters']['lam']
        self.alpha = self.local_config['parameters']['alpha']
        self.epochs = self.local_config['parameters']['epochs']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        

        self.model = ExplainModelGraph(self.n_nodes).to(self.device)
        self.model._fitted = False

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['batch_size_ratio'] =  self.local_config['parameters'].get('batch_size_ratio', 0.1)
        self.local_config['parameters']['lr'] =  self.local_config['parameters'].get('lr', 1e-3)
        self.local_config['parameters']['weight_decay'] =  self.local_config['parameters'].get('weight_decay', 0)
        self.local_config['parameters']['gamma'] =  self.local_config['parameters'].get('gamma', 1e-4)
        self.local_config['parameters']['lam'] =  self.local_config['parameters'].get('lam', 1e-4)
        self.local_config['parameters']['alpha'] =  self.local_config['parameters'].get('alpha', 1e-4)
        self.local_config['parameters']['epochs'] =  self.local_config['parameters'].get('epochs', 200)

        # fix the number of nodes
        n_nodes = self.local_config['parameters'].get('n_nodes', None)
        if not n_nodes:
            n_nodes = max([x.num_nodes for x in self.dataset.instances])
        self.local_config['parameters']['n_nodes'] = n_nodes

    def real_fit(self):
        self.model.train()

        for epoch in range(self.epochs):         
            losses = list()

            for graph in self.dataset.instances:
                pred1, pred2 = self.model(graph, self.oracle)
                loss = self.model.loss(graph,
                                           pred1, pred2,
                                           self.gamma, self.lam,
                                           self.alpha)
                
                losses.append(loss.to('cpu').detach().numpy())
                loss.backward()
                self.optimizer.step()
            self.context.logger.info(f"Epoch {epoch+1} --- loss {np.mean(losses)}")
        
        self.model._fitted = True

    def explain(self, instance : GraphInstance):

        if(not self.model._fitted):
            self.fit()

        self.model.eval()
        
        with torch.no_grad():
            cf_instance = deepcopy(instance)

            weighted_adj = self.model._rebuild_weighted_adj(instance)
            masked_adj = self.model.get_masked_adj(weighted_adj).numpy()
            # update instance copy from masked_ajd
            # cf_instance.data = masked_adj        

            new_adj = np.where(masked_adj != 0, 1, 0)
            # the weights need to be an array of real numbers with
            # length equal to the number of edges
            row_indices, col_indices = np.where(masked_adj != 0)
            weights = masked_adj[row_indices, col_indices]

            cf_instance.data = new_adj
            cf_instance.edge_weights = weights
            # avoid the old nx representation
            cf_instance._nx_repr = None
			
            return cf_instance


class ExplainModelGraph(torch.nn.Module):
    
    def __init__(self, n_nodes: int):
        super(ExplainModelGraph, self).__init__()

        self.n_nodes = n_nodes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mask = self.build_adj_mask()

    def forward(self, graph : GraphInstance, oracle : Oracle):        
        pred1 = oracle.predict(graph)

        # re-build weighted adjacency matrix
        weighted_adj = self._rebuild_weighted_adj(graph)
        # get the masked_adj
        masked_adj = self.get_masked_adj(weighted_adj)
        # get the new weights as the difference between
        # the weighted adjacency matrix and the masked learned
        new_weights = weighted_adj - masked_adj
        # get only the edges that exist
        row_indices, col_indices = torch.where(new_weights != 0)

        cf_instance = deepcopy(graph)
        cf_instance.edge_weights = new_weights[row_indices, col_indices].detach().numpy()
        # avoid old nx representation
        cf_instance._nx_repr = None
        pred2 = oracle.predict(cf_instance)

        pred1 = torch.Tensor([pred1]).float()  # factual
        pred2 = torch.Tensor([pred2]).float()  # counterfactual

        return pred1, pred2

    def build_adj_mask(self):
        mask = torch.nn.Parameter(torch.FloatTensor(self.n_nodes, self.n_nodes))
        std = torch.nn.init.calculate_gain("relu") * math.sqrt(
            2.0 / (self.n_nodes + self.n_nodes)
        )
        with torch.no_grad():
            mask.normal_(1.0, std)
        return mask

    def get_masked_adj(self, weights):
        sym_mask = torch.sigmoid(self.mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2
        masked_adj = weights * sym_mask
        return masked_adj

    def loss(self, graph : GraphInstance, pred1, pred2, gam, lam, alp):
        weights = self._rebuild_weighted_adj(graph)
        bpr1 = torch.nn.functional.relu(gam + 0.5 - pred1)  # factual
        bpr2 = torch.nn.functional.relu(gam + pred2 - 0.5)  # counterfactual
        masked_adj = torch.flatten(self.get_masked_adj(weights))
        L1 = torch.linalg.norm(masked_adj, ord=1)
        return L1 + lam * (alp * bpr1 + (1 - alp) * bpr2)
    
    
    # todo reimplement this part
    def _rebuild_weighted_adj(self, graph):
        weights = np.zeros((self.n_nodes, self.n_nodes))

        u = []
        v = []
        for i, j in zip(*np.nonzero(graph.data)):
            if i < j:
                u.append(i)
                v.append(j)
        #print(graph.edge_weights.shape)
        #print(graph.edge_weights)
        #print(u)
        #print(v)
        weights[u+v,v+u] = graph.edge_weights
        return torch.from_numpy(weights).float()