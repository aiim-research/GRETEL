import numpy as np
import random
import torch

from src.core.trainable_base import Trainable
from src.utils.cfg_utils import init_dflts_to_of
from src.core.factory_base import get_instance_kvargs
from sklearn.metrics import accuracy_score
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

class TorchBase(Trainable):
       
    def init(self):
        self.epochs = self.local_config['parameters']['epochs']
        self.batch_size = self.local_config['parameters']['batch_size']
        # Opt-in robust training (read without being written into the config, so the configs without it
        # keep their hash): {"drop": p, "add": q} removes each undirected edge of every training graph
        # with probability p and adds q times as many random edges, redrawn at every batch, so the model
        # learns to ignore a few edits and a counterfactual needs more of them.
        self.edge_perturbation = self.local_config['parameters'].get('edge_perturbation')
        
        self.model = get_instance_kvargs(self.local_config['parameters']['model']['class'],
                                   self.local_config['parameters']['model']['parameters'])
        
        self.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})
        
        self.loss_fn = get_instance_kvargs(self.local_config['parameters']['loss_fn']['class'],
                                           self.local_config['parameters']['loss_fn']['parameters'])
        
        self.early_stopping_threshold = self.local_config['parameters']['early_stopping_threshold']
        
        self.lr_scheduler =  lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5, total_iters=self.epochs)

        
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device) 
        
        self.patience = 0                           
    
    def real_fit(self):
              
        instances = self.dataset.get_torch_instances(fold_id=self.fold_id)
        train_loader, val_loader = None, None
        
        if self.early_stopping_threshold:
            num_instances = len(self.dataset.instances)
            # get 5% of training instances and reserve them for validation
            indices = list(range(num_instances))
            random.shuffle(indices)
            val_size = max(int(.05 * len(indices)), self.batch_size)
            train_size = len(indices) - val_size
            # get the training instances
            train_instances = Subset(instances, indices[:train_size - 1])
            val_instances = Subset(instances, indices[train_size:])
            # get the train and validation loaders
            train_loader = DataLoader(train_instances, batch_size=self.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_instances, batch_size=self.batch_size, shuffle=True, drop_last=True)
        else:
            train_loader = DataLoader(instances, batch_size=self.batch_size, shuffle=True, drop_last=True)

        best_loss = [0,0]
        
        for epoch in range(self.epochs):
            losses, preds, labels_list = [], [], []
            self.model.train()
            for batch in train_loader:
                batch.batch = batch.batch.to(self.device)
                node_features = batch.x.to(self.device)
                edge_index = batch.edge_index.to(self.device)
                edge_weights = batch.edge_attr.to(self.device)
                labels = batch.y.to(self.device).long()
                if self.edge_perturbation:
                    edge_index, edge_weights = perturb_edges(edge_index, edge_weights, batch.ptr.to(self.device),
                                                             **self.edge_perturbation)

                self.optimizer.zero_grad()
                
                pred = self.model(node_features, edge_index, edge_weights, batch.batch)
                loss = self.loss_fn(pred, labels)
                losses.append(loss.to('cpu').detach().numpy())
                loss.backward()
                
                labels_list += list(labels.squeeze().long().detach().to('cpu').numpy())
                preds += list(pred.squeeze().detach().to('cpu').numpy())
               
                self.optimizer.step()

            accuracy = self.accuracy(labels_list, preds)
            self.context.logger.info(f'epoch = {epoch} ---> loss = {np.mean(losses):.4f}\t accuracy = {accuracy:.4f}')
            self.lr_scheduler.step()
            
            # check if we need to do early stopping
            if self.early_stopping_threshold and len(val_loader) > 0:
                self.model.eval()
                var_losses, var_labels, var_preds = [], [], []
                with torch.no_grad():
                    for batch in val_loader:
                        batch.batch = batch.batch.to(self.device)
                        node_features = batch.x.to(self.device)
                        edge_index = batch.edge_index.to(self.device)
                        edge_weights = batch.edge_attr.to(self.device)
                        labels = batch.y.to(self.device).long()

                        pred = self.model(node_features, edge_index, edge_weights, batch.batch)
                        loss = self.loss_fn(pred, labels)
                        
                        var_labels += list(labels.squeeze().to('cpu').numpy())
                        var_preds += list(pred.squeeze().to('cpu').numpy())
                        
                        var_losses.append(loss.item())
                        
                    best_loss.pop(0)
                    var_loss = np.mean(var_losses)
                    best_loss.append(var_loss)
                            
                    accuracy = self.accuracy(var_labels, var_preds)
                    self.context.logger.info(f'epoch = {epoch} ---> var_loss = {var_loss:.4f}\t var_accuracy = {accuracy:.4f}')
                
                if abs(best_loss[0] - best_loss[1]) < self.early_stopping_threshold:
                    self.patience += 1
                    
                    if self.patience == 4:
                        self.context.logger.info(f"Early stopped training at epoch {epoch}")
                        break  # terminate the training loop
                
    def check_configuration(self):
        super().check_configuration()
        local_config=self.local_config
        # set defaults
        local_config['parameters']['epochs'] = local_config['parameters'].get('epochs', 200)
        # local_config['parameters']['batch_size'] = local_config['parameters'].get('batch_size', 4)
        local_config['parameters']['batch_size'] = local_config['parameters'].get('batch_size', 1)
        local_config['parameters']['early_stopping_threshold'] = local_config['parameters'].get('early_stopping_threshold', None)
        # populate the optimizer
        init_dflts_to_of(local_config, 'optimizer', 'torch.optim.Adam',lr=0.001)
        init_dflts_to_of(local_config, 'loss_fn', 'torch.nn.BCELoss')
        
    def accuracy(self, testy, probs):
        acc = accuracy_score(testy, np.argmax(probs, axis=1))
        return acc

    def read(self):
        super().read()
        if isinstance(self.model, list):
            for mod in self.model:
                mod.to(self.device)
        else:
            self.model.to(self.device)
            
    def to(self, device):
        if isinstance(self.model, torch.nn.Module):
            self.model.to(device)
        elif isinstance(self.model, list):
            for model in self.model:
                if isinstance(model, torch.nn.Module):
                    model.to(self.device)


def perturb_edges(edge_index, edge_weights, ptr, drop=0.0, add=0.0):
    """Random symmetric edge perturbation of a batch of undirected graphs (each edge stored in both
    directions): every edge is kept with probability 1 - drop, and round(add * number of edges) random
    edges are added inside random graphs of the batch, with weight 1."""
    upper = edge_index[0] < edge_index[1]
    pairs, weights = edge_index[:, upper], edge_weights[upper]
    n_edges = pairs.shape[1]
    if drop > 0:
        keep = torch.rand(n_edges, device=pairs.device) >= drop
        pairs, weights = pairs[:, keep], weights[keep]
    n_add = int(round(add * n_edges))
    if n_add > 0:
        n_graphs = ptr.numel() - 1
        g = torch.randint(0, n_graphs, (n_add,), device=pairs.device)
        start, size = ptr[g], ptr[g + 1] - ptr[g]
        u = start + (torch.rand(n_add, device=pairs.device) * size).long()
        v = start + (torch.rand(n_add, device=pairs.device) * size).long()
        ok = u != v
        new = torch.stack([torch.minimum(u, v), torch.maximum(u, v)])[:, ok]
        pairs = torch.cat([pairs, new], 1)
        weights = torch.cat([weights, torch.ones(new.shape[1], dtype=weights.dtype, device=weights.device)])
    pairs, idx = torch.unique(pairs, dim=1, return_inverse=True)
    w = torch.zeros(pairs.shape[1], dtype=weights.dtype, device=weights.device).scatter_(0, idx, weights)
    return torch.cat([pairs, pairs.flip(0)], 1), torch.cat([w, w])
