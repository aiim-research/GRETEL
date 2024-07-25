import copy
import itertools
import math
import numpy as np
import torch
from typing import Any, Tuple

from src.core.factory_base import get_instance_kvargs
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric
from src.explainer.generative.gans.graph.learnable_edges.graph_embedders import GraphEmbedder
from src.explainer.generative.gans.model import BaseGAN
from src.dataset.instances.graph import GraphInstance
from src.utils.torch.utils import rebuild_adj_matrix
from src.dataset.utils.dataset_torch import TorchGeometricDataset

from torch_geometric.utils import unbatch, unbatch_edge_index
from src.utils.cfg_utils import init_dflts_to_of

class EdgeLearnableGAN(BaseGAN):

    def init(self):
        super().init()
        local_params = self.local_config['parameters']
        self.edge_module = get_instance_kvargs(local_params['edge_module']['class'],
                                               local_params['edge_module']['parameters'])
        
        self.edge_optimizer = get_instance_kvargs(local_params['edge_optimizer']['class'],
                                                  {'params':self.edge_module.parameters(), 
                                                   **local_params['edge_optimizer']['parameters']})
        

        self.node_module = get_instance_kvargs(local_params['node_module']['class'],
                                                     local_params['node_module']['parameters'])
        
        self.node_optimizer = get_instance_kvargs(local_params['node_optimizer']['class'],
                                                       {'params':self.node_module.parameters(), 
                                                        **local_params['node_optimizer']['parameters']})
        
        self.alpha = local_params.get('alpha', .5)
        self.tau = local_params.get('tau', .5)

        self.graph_embedder = GraphEmbedder(num_nodes=np.max(self.dataset.num_nodes_values),
                                            node_feature_dim=self.dataset.num_node_features(), 
                                            dim=local_params['generator']['parameters'].get('out_embed_dim', 4))
                
        self.edge_module.to(torch.double)        
        self.edge_module.to(self.device)
        self.edge_module.device = self.device

        self.node_module.to(torch.double)        
        self.node_module.to(self.device)
        self.node_module.device = self.device

        self.rec_loss_fn = torch.nn.MSELoss()
        self.edge_loss_fn = torch.nn.BCEWithLogitsLoss()
        self.ged = GraphEditDistanceMetric()

        for p in self.edge_module.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -.1, .1))
    
    def infinite_data_stream(self, loader):
        # Define a generator function that yields batches of data
        while True:
            for batch in loader:
                yield batch.to(self.device)
                
    def real_fit(self):   
        discriminator_loader = self.dataset.get_torch_loader(fold_id=self.fold_id,
                                                             batch_size=self.batch_size,
                                                             kls=self.explainee_label)
        
        generator_loader =  self.dataset.get_torch_loader(fold_id=self.fold_id,
                                                          batch_size=self.batch_size,
                                                          kls=1-self.explainee_label)

        all_edges = torch.tensor(list(itertools.product(range(self.dataset.num_nodes), repeat=2))).T  # (2, d)

        torch.autograd.set_detect_anomaly(True)
        
        for epoch in range(self.epochs):
            losses_G, losses_D, rec_losses, edge_losses = [], [], [], []

            for cf_data, f_data in itertools.zip_longest(discriminator_loader, generator_loader, fillvalue=None):

                if cf_data is not None:
                    cf_node_features, cf_edge_index, cf_edge_features, _, cf_batch, _ = cf_data
                    
                    real_labels = torch.ones(self.batch_size, 1).to(self.device)
                    fake_labels = torch.zeros(self.batch_size, 1).to(self.device)
                    #######################################################################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    self.discriminator_optimizer.zero_grad()
                    # discriminator on real data (counterfactuals)
                    cf_emb_nodes = self.graph_embedder(cf_node_features[1], cf_edge_index[1], cf_edge_features[1], cf_batch[1])
                    cf_emb_nodes = torch.stack(unbatch(cf_emb_nodes, cf_batch[1]))
                    output_real = self.discriminator(cf_emb_nodes)
                    loss_real = self.loss_fn(output_real.double(), real_labels.double())
                    # learn how to reconstruct the counterfactual node embeddings
                    self.node_optimizer.zero_grad()
                    cf_recon_nodes = self.node_module(cf_emb_nodes)
                    rec_loss = self.rec_loss_fn(torch.stack(unbatch(cf_node_features[1], cf_batch[1])), cf_recon_nodes)
                    rec_losses.append(rec_loss.item())
                    rec_loss.backward(retain_graph=True)
                    self.node_optimizer.step()
                    # learn how to estimate edges
                    self.edge_optimizer.zero_grad()
                    _, cf_edge_logits, cf_true_edges = self.edge_module(cf_emb_nodes, cf_edge_index[1], cf_batch[1])
                    edge_loss = self.edge_loss_fn(cf_true_edges.double(), cf_edge_logits.double())
                    edge_losses.append(edge_loss.item())
                    edge_loss.backward()
                    self.edge_optimizer.step()

                    if f_data is not None:
                        f_node_features, f_edge_index, f_edge_features, _ , f_batch , _ = f_data
                        # generate the node embeddings
                        f_emb_nodes = self.generator(f_node_features[1], f_edge_index[1],
                                                     f_edge_features[1], f_batch[1])
                        
                        f_emb_nodes = torch.stack(unbatch(f_emb_nodes, f_batch[1]))
                    
                        output_fake = self.discriminator(f_emb_nodes)
                        loss_fake = self.loss_fn(output_fake.double(), fake_labels.double())

                        loss_D = loss_real + loss_fake

                        # reparametrize the edges the fake instances
                        f_recon_nodes = self.node_module(f_emb_nodes)
                        _, f_edge_logits, _ = self.edge_module(f_emb_nodes, f_edge_index[1], f_batch[1])
                        fake_inst = []
                        for batch_idx, edge_logits in enumerate(f_edge_logits):
                            f_repr_edge_index, f_repr_edge_weight = self.__reparametrization_trick(all_edges, edge_logits, self.tau)
                            fake_inst += self.retake_batch(f_recon_nodes[batch_idx], f_repr_edge_index, f_repr_edge_weight, f_batch[1], counterfactual=True, generator=True)
                        y_batch = torch.cat((real_labels, fake_labels))
                    else:
                        loss_D = loss_real
                        fake_inst = []
                        y_batch = real_labels

                    y_batch = y_batch.squeeze()
                    # take the oracle's prediction for the true instances
                    real_inst = []
                    for batch_idx, edge_logits in enumerate(cf_edge_logits):
                        cf_repr_edge_index, cf_repr_edge_weight = self.__reparametrization_trick(all_edges, edge_logits, self.tau)
                        real_inst += self.retake_batch(cf_recon_nodes[batch_idx], cf_repr_edge_index, cf_repr_edge_weight, cf_batch[1])
                    # oracle scores    
                    oracle_scores = self.take_oracle_predictions(real_inst + fake_inst, y_batch)
                    # gradient descent on the discriminator
                    loss_D = torch.mean(loss_D * torch.tensor(oracle_scores, dtype=torch.float))
                    losses_D.append(loss_D.item())
                    loss_D.backward()
                    self.discriminator_optimizer.step()
                    #######################################################################

                if f_data is not None:
                    self.generator_optimizer.zero_grad()

                    f_node_features, f_edge_index, f_edge_features, _ , f_batch , _ = f_data
                    y_fake = torch.ones(self.batch_size, 1).to(self.device)
                    # generate the node embeddings
                    f_emb_nodes = self.generator(f_node_features[1], f_edge_index[1],
                                                 f_edge_features[1], f_batch[1])
                    
                    f_emb_nodes = torch.stack(unbatch(f_emb_nodes, f_batch[1]))
                    output_fake = self.discriminator(f_emb_nodes)

                    loss_G = torch.mean(self.loss_fn(output_fake.double(), y_fake.double()))
                    losses_G.append(loss_G.item())
                    loss_G.backward()
                    self.generator_optimizer.step()

                # Print the losses
            self.context.logger.info(f'Epoch [{epoch+1}/{self.epochs}], Loss D: {np.mean(losses_D)}, Loss G: {np.mean(losses_G)}, Loss Edge: {np.mean(edge_losses)}, Loss Nodes: {np.mean(rec_losses)}')
            

    def retake_batch(self, node_features, edge_indices, edge_features, batch, counterfactual=False, generator=False):
        return [GraphInstance(id="dummy",
                              label=1-self.explainee_label if counterfactual else self.explainee_label,
                              data=rebuild_adj_matrix(len(node_features), edge_indices, edge_features.T, self.device).detach().cpu().numpy(),
                              node_features=node_features.detach().cpu().numpy(),
                              edge_features=edge_features.detach().cpu().numpy())]

    
    def __gumbel_sigmoid_sample(self, probabilities, tau, eps=1e-10):
        """
        Apply Gumbel-Softmax trick to binary variables (edges).
        Arguments:
        - probabilities: tensor of shape (d,) containing probabilities of edges.
        - tau: temperature parameter.
        Returns:
        - Sampled tensor of shape (d,) with gradients.
        """
        def gumbel_sample(shape, eps=1e-20):
            """Sample from Gumbel(0, 1)"""
            U = torch.rand(shape)
            return -torch.log(-torch.log(U + eps) + eps)
        
        probabilities = torch.clamp(probabilities, eps, 1 - eps)  # Clamp probabilities to avoid log(0)
        logits = torch.log(probabilities) - torch.log(1 - probabilities)
        gumbel_noise = gumbel_sample(probabilities.size())
        return torch.sigmoid((logits + gumbel_noise) / tau)
    

    def __reparametrization_trick(self, all_edges, edge_probs, tau):
        # Sampling without breaking gradiesnts
        sampled_probs = self.__gumbel_sigmoid_sample(edge_probs, tau)
        # Generating binary samples based on the sampled probabilities
        sampled_edges = torch.bernoulli(sampled_probs)
        # Create the (2, d) tensor where each column represents an edge with the first and second node
        edge_index = all_edges * sampled_edges
        # Filter out zero columns (edges not sampled)
        edge_index = edge_index[:, sampled_edges.bool()]
        return edge_index.long(), sampled_probs[sampled_edges.bool()]
    
    def check_configuration(self):
        dflt_generator = "src.explainer.generative.gans.graph.learnable_edges.generators.TranslatingGenerator"
        dflt_discriminator =  "src.explainer.generative.gans.graph.learnable_edges.discriminators.EmbeddingDiscriminator"
        dflt_edge_module = "src.explainer.generative.gans.graph.learnable_edges.graph_embedders.EdgeExistanceModule"
        dflt_node_module = "src.explainer.generative.gans.graph.learnable_edges.graph_embedders.NodeDecoderModule"

        if 'discriminator' in self.local_config['parameters']\
            and 'parameters' in self.local_config['parameters']['discriminator']:
            dropout = self.local_config['parameters']['discriminator']['parameters'].get('dropout', .2)
        else:
            dropout = .2

        if 'generator' in self.local_config['parameters']\
            and 'parameters' in self.local_config['parameters']['generator']:
            in_embed_dim = self.local_config['parameters']['generator']['parameters'].get('in_embed_dim', 10)
            out_embed_dim = self.local_config['parameters']['generator']['parameters'].get('out_embed_dim', 4)
            num_translator_layers = self.local_config['parameters']['generator']['parameters'].get('num_translator_layers', 4)
            gaussian_std = self.local_config['parameters']['generator']['parameters'].get('gaussian_std', .1)
        else:
            in_embed_dim, out_embed_dim, num_translator_layers, gaussian_std = 10, 4, 4, .1

        
        #Check if the generator exist or build with its defaults:
        init_dflts_to_of(self.local_config, 
                         'generator',
                         dflt_generator,
                         k=self.dataset.num_nodes,
                         in_embed_dim=in_embed_dim,
                         out_embed_dim=out_embed_dim,
                         num_translator_layers=num_translator_layers,
                         node_features=self.dataset.num_node_features(),
                         gaussian_std=gaussian_std)
        #Check if the discriminator exist or build with its defaults:
        init_dflts_to_of(self.local_config,
                         'discriminator',
                         dflt_discriminator,
                         num_nodes=np.max(self.dataset.num_nodes_values),
                         dropout=dropout,
                         dim=out_embed_dim)  
        # Check if the edge module exists or build with its defaults:
        init_dflts_to_of(self.local_config,
                         'edge_module',
                         dflt_edge_module,
                         dim=out_embed_dim)
        
        # Check if the node module exists or build with its defaults:
        init_dflts_to_of(self.local_config,
                         'node_module',
                         dflt_node_module,
                         num_nodes=np.max(self.dataset.num_nodes_values),
                         node_feature_dim=self.dataset.num_node_features(),
                         dim=out_embed_dim)
        
        # If the gen_optimizer is not present we create it
        if 'node_optimizer' not in self.local_config['parameters']:
            init_dflts_to_of(self.local_config, 'node_optimizer','torch.optim.SGD',lr=0.001)

        if 'edge_optimizer' not in self.local_config['parameters']:
            init_dflts_to_of(self.local_config, 'edge_optimizer','torch.optim.SGD',lr=0.001)
        
        super().check_configuration()
        
    def __call__(self, *args: Tuple[GraphInstance], **kwds: Any) -> Any:
        with torch.no_grad():
            batch = TorchGeometricDataset.to_geometric(args[0]).to(self.device)
            node_emb = self.generator(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            rec_nodes = self.node_module(node_emb)
            _, edge_probs, _ = self.edge_module(node_emb, batch.edge_index)
            return rec_nodes, edge_probs   