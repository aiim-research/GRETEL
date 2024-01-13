from typing import Any, Tuple
import numpy as np
import torch

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from src.explainer.generative.gans.model import BaseGAN
from src.dataset.instances.graph import GraphInstance
from src.utils.cfg_utils import init_dflts_to_of


class GAN(BaseGAN):
                
    def infinite_data_stream(self, loader):
        # Define a generator function that yields batches of data
        while True:
            for adj, label, node_features in loader:
                yield adj.to(self.device), label, node_features.to(self.device)
                
    def real_fit(self):
        max_nodes = max(self.dataset.num_nodes_values)
        discriminator_loader = DataLoader(
            self.dataset.get_torch_instances(fold_id=self.fold_id,
                                             kls=self.explainee_label,
                                             dataset_kls='src.dataset.utils.dataset_torch.TorchDataset',
                                             max_nodes=max_nodes), 
            batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        generator_loader = DataLoader(
            self.dataset.get_torch_instances(fold_id=self.fold_id,
                                             kls=1-self.explainee_label,
                                             dataset_kls='src.dataset.utils.dataset_torch.TorchDataset',
                                             max_nodes=max_nodes), 
            batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        discriminator_loader = self.infinite_data_stream(discriminator_loader)
        # TODO: make it multiclass in Dataset
        generator_loader = self.infinite_data_stream(generator_loader)

        for epoch in range(self.epochs):
            G_losses, D_losses = [], []
            #######################################################################
            self.prepare_discriminator_for_training()
            #######################################################################
            # discriminator data (real batch)
            f_graph, f_node_features, _ = next(discriminator_loader)
            f_graph = f_graph.double().to(self.device)[:,None,:,:]
            # generator data (fake batch)
            cf_graph, cf_node_features, _  = next(generator_loader)
            cf_graph = cf_graph.double().to(self.device)[:,None,:,:]
            
            cf_graph = self.generator(cf_graph).double()
            # get the real and fake labels
            y_batch = torch.cat([torch.ones(self.batch_size,), torch.zeros(self.batch_size,)], dim=0).to(self.device)
            #######################################################################
            # get the oracle's predictions
            real_inst = self.retake_batch(f_graph, f_node_features)
            fake_inst = self.retake_batch(cf_graph, cf_node_features, counterfactual=True)
            oracle_scores = self.take_oracle_predictions(real_inst + fake_inst, y_batch)
            #######################################################################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            real_pred = self.discriminator(f_graph)
            fake_pred = self.discriminator(cf_graph)
            y_pred = torch.cat([real_pred, fake_pred])
            loss = torch.mean(self.loss_fn(y_pred.squeeze().double(), y_batch.double()) * torch.tensor(oracle_scores, dtype=torch.float))
            D_losses.append(loss.item())
            loss.backward()
            self.discriminator_optimizer.step()
            #######################################################################
            self.prepare_generator_for_training()
            ## Update G network: maximize log(D(G(z)))
            cf_graph, _, _ = next(generator_loader)
            cf_graph = cf_graph.double().to(self.device)[:,None,:,:]
            y_fake = torch.ones((self.batch_size, 1)).to(self.device)
            output = self.discriminator(self.generator(cf_graph).double())
            # calculate the loss
            loss = self.loss_fn(output.double(), y_fake.double())
            loss.backward()
            G_losses.append(loss.item())
            self.generator_optimizer.step()
                
            self.context.logger.info(f'Epoch {epoch}\t Loss_D = {np.mean(D_losses): .4f}\t Loss_G = {np.mean(G_losses): .4f}')
  
    def check_configuration(self):
        dflt_generator = 'src.explainer.generative.gans.image.generators.ResGenerator'
        dflt_discriminator = 'src.explainer.generative.gans.image.discriminators.SimpleDiscriminator'

        #Check if the generator exist or build with its defaults:
        init_dflts_to_of(self.local_config, 'generator', dflt_generator, 
                         num_nodes=max(self.dataset.num_nodes_values))
        #Check if the generator exist or build with its defaults:
        init_dflts_to_of(self.local_config, 'discriminator', dflt_discriminator, 
                         num_nodes=max(self.dataset.num_nodes_values))
        
        super().check_configuration()
        
        
    def retake_batch(self, graph: torch.Tensor, node_features: torch.Tensor, counterfactual=True):
        # create the instances
        instances = []
        for i in range(len(graph)):
            instances.append(GraphInstance(id="dummy",
                                           label=1-self.explainee_label if counterfactual else self.explainee_label,
                                           data=graph[i].squeeze().cpu().numpy(),
                                           node_features=node_features[i].cpu().numpy()))
        return instances    
    
    def __call__(self, *args: Tuple[GraphInstance], **kwds: Any) -> Any:
        torch_data = torch.from_numpy(args[0].data[None,None,:,:]).double()
        return self.generator(torch_data)