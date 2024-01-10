import math
import numpy as np
import torch
from typing import Any, Tuple

from src.explainer.generative.gans.model import BaseGAN
from src.dataset.instances.graph import GraphInstance
from src.utils.torch.utils import rebuild_adj_matrix
from src.dataset.utils.dataset_torch import TorchGeometricDataset

from torch_geometric.utils.unbatch import unbatch, unbatch_edge_index
from src.utils.cfg_utils import init_dflts_to_of

class GAN(BaseGAN):
    
    def infinite_data_stream(self, loader):
        # Define a generator function that yields batches of data
        while True:
            for batch in loader:
                yield batch.to(self.device)
                
    def real_fit(self):
        discriminator_loader = self.infinite_data_stream(self.dataset.get_torch_loader(fold_id=self.fold_id, batch_size=self.batch_size, kls=self.explainee_label))
        # TODO: make it multiclass in Datase
        generator_loader = self.infinite_data_stream(self.dataset.get_torch_loader(fold_id=self.fold_id, batch_size=self.batch_size, kls=1-self.explainee_label))

        for epoch in range(self.epochs):
            G_losses, D_losses = [], []
            #######################################################################
            self.prepare_discriminator_for_training()
            #######################################################################
            # discriminator data (real batch)
            node_features, edge_index, edge_features, _ , real_batch ,_ = next(discriminator_loader)
            # generator data (fake batch)
            fake_node_features, fake_edge_index, fake_edge_features, _ , fake_batch , _  = next(generator_loader)
            _, fake_edge_index, fake_edge_probs = self.generator(fake_node_features[1], fake_edge_index[1], fake_edge_features[1], fake_batch[1])
            # get the real and fake labels
            y_batch = torch.cat([torch.ones((len(torch.unique(real_batch[1])),)),
                                 torch.zeros(len(torch.unique(fake_batch[1])),)], dim=0).to(self.device)
            #######################################################################
            # get the oracle's predictions
            real_inst = self.retake_batch(node_features[1], edge_index[1], edge_features[1], real_batch[1])
            fake_inst = self.retake_batch(fake_node_features[1], fake_edge_index, fake_edge_probs, fake_batch[1], counterfactual=True, generator=True)
            oracle_scores = self.take_oracle_predictions(real_inst + fake_inst, y_batch)
            #######################################################################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            real_pred = self.discriminator(node_features[1], edge_index[1], edge_features[1]).expand(1)
            fake_pred = self.discriminator(fake_node_features[1], fake_edge_index, fake_edge_features[1]).expand(1)
            y_pred = torch.cat([real_pred, fake_pred])
            loss = torch.mean(self.loss_fn(y_pred.squeeze().double(), y_batch.double()) * torch.tensor(oracle_scores, dtype=torch.float))
            D_losses.append(loss.item())
            loss.backward()
            self.discriminator_optimizer.step()
            #######################################################################
            self.prepare_generator_for_training()
            ## Update G network: maximize log(D(G(z)))
            fake_features, fake_edge_index, fake_edge_attr, _, fake_batch, _ = next(generator_loader)
            y_fake = torch.ones((len(torch.unique(fake_batch[1])),)).to(self.device)
            output = self.discriminator(self.generator(fake_features[1], fake_edge_index[1], fake_edge_attr[1], fake_batch[1])[0], fake_edge_index[1], fake_edge_attr[1])
            # calculate the loss
            loss = self.loss_fn(output.expand(1).double(), y_fake.double())
            loss.backward()
            G_losses.append(loss.item())
            self.generator_optimizer.step()
                
            self.context.logger.info(f'Epoch {epoch}\t Loss_D = {np.mean(D_losses): .4f}\t Loss_G = {np.mean(G_losses): .4f}')
  
    def retake_batch(self, node_features, edge_indices, edge_features, batch, counterfactual=False, generator=False):
        # unbatch edge indices
        edges = unbatch_edge_index(edge_indices, batch)
        # unbatch node_features
        node_features = unbatch(node_features, batch)
        # unbatch edge features
        if not generator:
            sizes = [index.shape[-1] for index in edges]
            edge_features = edge_features.split(sizes)
        # create the instances
        instances = []
        for i in range(len(edges)):
            if not generator:
                unbatched_edge_features = edge_features[i]
            else:
                mask = torch.zeros(edge_features.shape).to(self.device)
                mask[edges[i][0,:], edges[i][1,:]] = 1
                unbatched_edge_features = edge_features * mask
                indices = torch.nonzero(unbatched_edge_features)
                unbatched_edge_features = unbatched_edge_features[indices[:,0], indices[:,1]]
                
            instances.append(GraphInstance(id="dummy",
                                           label=1-self.explainee_label if counterfactual else self.explainee_label,
                                           data=rebuild_adj_matrix(len(node_features[i]), edges[i], unbatched_edge_features.T, self.device).detach().cpu().numpy(),
                                           node_features=node_features[i].detach().cpu().numpy(),
                                           edge_features=unbatched_edge_features.detach().cpu().numpy()))
        return instances
    
    def check_configuration(self):
        dflt_generator = "src.explainer.generative.gans.graph.res_gen.ResGenerator"
        dflt_discriminator =  "src.explainer.generative.gans.graph.discriminators.TopKPoolingDiscriminator" #"src.explainer.generative.gans.graph.discriminators.SimpleDiscriminator"
        
        sqrt_features = int(math.sqrt(self.dataset.num_node_features())) + 1
        if 'discriminator' in self.local_config['parameters']\
            and 'parameters' in self.local_config['parameters']['discriminator']:
            embed_dim_discr = self.local_config['parameters']['discriminator']['parameters'].get('embed_dim', sqrt_features)
        else:
            embed_dim_discr = sqrt_features
        #Check if the generator exist or build with its defaults:
        init_dflts_to_of(self.local_config, 
                         'generator',
                         dflt_generator,
                         node_features=self.dataset.num_node_features())
        #Check if the generator exist or build with its defaults:
        init_dflts_to_of(self.local_config,
                         'discriminator',
                         dflt_discriminator,
                         num_nodes=self.dataset.num_nodes,
                         node_features=self.dataset.num_node_features(),
                         dim=embed_dim_discr)  
        
        super().check_configuration()
        
    def __call__(self, *args: Tuple[GraphInstance], **kwds: Any) -> Any:
        batch = TorchGeometricDataset.to_geometric(args[0]).to(self.device)
        return self.generator(batch.x, batch.edge_index, batch.edge_attr, batch.batch)