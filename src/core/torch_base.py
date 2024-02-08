import numpy as np
import torch
from src.core.trainable_base import Trainable
from src.utils.cfg_utils import init_dflts_to_of
from src.core.factory_base import get_instance_kvargs
from sklearn.metrics import accuracy_score
import torch.optim.lr_scheduler as lr_scheduler

import optuna
from src.oracle.nn.gcn import DownstreamGCN

class TorchBase(Trainable):
       
    def init(self):
        self.epochs = self.local_config['parameters']['epochs']
        self.batch_size = self.local_config['parameters']['batch_size']
        self.optimize_hyperparameters_GCN = self.local_config['parameters']['optimize_hyperparameters_GCN']
        
        self.model = get_instance_kvargs(self.local_config['parameters']['model']['class'],
                                   self.local_config['parameters']['model']['parameters'])
        
        self.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})
        
        self.loss_fn = get_instance_kvargs(self.local_config['parameters']['loss_fn']['class'],
                                           self.local_config['parameters']['loss_fn']['parameters'])
        
        self.lr_scheduler =  lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5, total_iters=self.epochs)

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device)                            
    
    def real_fit(self):
        if self.optimize_hyperparameters_GCN:
            self.context.logger.info("Optimizing hyperparameters")
            get_instance_kvargs(self.local_config['parameters']['model']['class'],
                                self.get_best_hyperparameters())
            
            self.context.logger.info("Retraining the best found model")
            loader = self.dataset.get_torch_loader(fold_id=self.fold_id, batch_size=self.batch_size, usage='train')
            
            for epoch in range(self.epochs):
                losses = []
                preds = []
                labels_list = []
                for batch in loader:
                    batch.batch = batch.batch.to(self.device)
                    node_features = batch.x.to(self.device)
                    edge_index = batch.edge_index.to(self.device)
                    edge_weights = batch.edge_attr.to(self.device)
                    labels = batch.y.to(self.device).long()
                    
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
        else:
            self.context.logger.info("Training the model without hyperparameter optimization")
            loader = self.dataset.get_torch_loader(fold_id=self.fold_id, batch_size=self.batch_size, usage='train')
            
            for epoch in range(self.epochs):
                losses = []
                preds = []
                labels_list = []
                for batch in loader:
                    batch.batch = batch.batch.to(self.device)
                    node_features = batch.x.to(self.device)
                    edge_index = batch.edge_index.to(self.device)
                    edge_weights = batch.edge_attr.to(self.device)
                    labels = batch.y.to(self.device).long()
                    
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
            
    def check_configuration(self):
        super().check_configuration()
        local_config=self.local_config
        # set defaults
        local_config['parameters']['epochs'] = local_config['parameters'].get('epochs', 200)
        local_config['parameters']['batch_size'] = local_config['parameters'].get('batch_size', 4)
        local_config['parameters']['optimize_hyperparameters_GCN'] = local_config['parameters'].get('optimize_hyperparameters_GCN', False)
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

    # hyperparameter search related functions
    def optuna_objective_GCN(self, trial):
        num_conv_layers = trial.suggest_int("num_conv_layers",2,10,log=True)
        num_dense_layers = trial.suggest_int("num_dense_layers",1,5,log=True)
        conv_booster = trial.suggest_float("conv_booster",0,3)
        linear_decay = trial.suggest_float("linear_decay",0,3)
        model_parameters = {"num_conv_layers": num_conv_layers, "num_dense_layers": num_dense_layers, "conv_booster": conv_booster, "linear_decay": linear_decay, "node_features": 7, "n_classes": 2}
        model = get_instance_kvargs(self.local_config['parameters']['model']['class'],
                                     model_parameters)
        
        optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params': model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})
        
        loss_fn = get_instance_kvargs(self.local_config['parameters']['loss_fn']['class'],
                                           self.local_config['parameters']['loss_fn']['parameters'])
        
        scheduler =  lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=self.epochs)

        loader = self.dataset.get_torch_loader(fold_id=self.fold_id, batch_size=self.batch_size, usage='train')
            
        for epoch in range(self.epochs):
            losses = []
            preds = []
            labels_list = []
            for batch in loader:
                batch.batch = batch.batch.to(self.device)
                node_features = batch.x.to(self.device)
                edge_index = batch.edge_index.to(self.device)
                edge_weights = batch.edge_attr.to(self.device)
                labels = batch.y.to(self.device).long()
                
                optimizer.zero_grad()
                
                pred = model(node_features, edge_index, edge_weights, batch.batch)
                loss = loss_fn(pred, labels)
                losses.append(loss.to('cpu').detach().numpy())
                loss.backward()
                
                labels_list += list(labels.squeeze().long().detach().to('cpu').numpy())
                preds += list(pred.squeeze().detach().to('cpu').numpy())
            
                optimizer.step()

            accuracy = self.accuracy(labels_list, preds)
            self.context.logger.info(f'epoch = {epoch} ---> loss = {np.mean(losses):.4f}\t accuracy = {accuracy:.4f}')
            scheduler.step()
        print(f"mean loss: {np.mean(losses)}")
        return np.mean(losses)
    
    def get_best_hyperparameters(self):
        study = optuna.create_study(study_name="GCN optimization")
        study.optimize(self.optuna_objective_GCN, n_trials=15)
        self.context.logger.info(f"Best hyperparamteres found: {study.best_params}")
        return {"num_conv_layers": study.best_params['num_conv_layers'], "num_dense_layers": study.best_params['num_dense_layers'], "conv_booster":study.best_params['conv_booster'], "linear_decay":study.best_params['linear_decay'], "node_features": 7, "n_classes": 2}
