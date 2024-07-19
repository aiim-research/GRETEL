import copy
import math
import numpy as np
import torch
from typing import Any, Tuple

from src.core.factory_base import get_instance_kvargs
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

        self.graph_embedder = GraphEmbedder(num_nodes=self.dataset.num_nodes,
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
    
    def infinite_data_stream(self, loader):
        # Define a generator function that yields batches of data
        while True:
            for batch in loader:
                yield batch.to(self.device)
                
    def real_fit(self):        
        discriminator_loader = self.infinite_data_stream(self.dataset.get_torch_loader(fold_id=self.fold_id,
                                                                                       batch_size=self.batch_size,
                                                                                       kls=self.explainee_label))
        generator_loader = self.infinite_data_stream(self.dataset.get_torch_loader(fold_id=self.fold_id,
                                                                                   batch_size=self.batch_size,
                                                                                   kls=1-self.explainee_label))
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.epochs):
            G_losses, D_losses, rec_losses, edge_losses = [], [], [], []
            self.prepare_discriminator_for_training()
            #######################################################################
            # discriminator data (real batch)
            cf_node_features, cf_edge_index, cf_edge_features, _ , cf_batch ,_ = next(discriminator_loader)
            # generator data (fake batch)
            f_node_features, f_edge_index, f_edge_features, _ , f_batch , _  = next(generator_loader)
            # generate the node embeddings
            f_embedded_nodes = self.generator(f_node_features[1],
                                              f_edge_index[1],
                                              f_edge_features[1],
                                              f_batch[1])
            # get the real and fake labels
            y_batch = torch.cat([torch.ones((len(torch.unique(cf_batch[1])),)),
                                 torch.zeros(len(torch.unique(f_batch[1])),)], dim=0).to(self.device)     
            #######################################################################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            f_pred = self.discriminator(f_embedded_nodes)
            # embed the nodes
            cf_emb_nodes = self.graph_embedder(cf_node_features[1], cf_edge_index[1], cf_edge_features[1], cf_batch[1])
            cf_pred = self.discriminator(cf_emb_nodes)
            # backward loss on the discriminator
            y_pred = torch.cat([cf_pred, f_pred], dim=0)
            d_loss = torch.mean(self.loss_fn(y_pred.double(), y_batch.double()))
            D_losses.append(d_loss.item())
            d_loss.backward(retain_graph=True)
            self.discriminator_optimizer.step()

            # update the node reconstruction module
            self.node_optimizer.zero_grad()
            cf_recon_nodes = self.node_module(cf_emb_nodes)
            rec_loss = self.alpha * self.rec_loss_fn(cf_node_features[1], cf_recon_nodes)
            rec_losses.append(rec_loss.item())
            rec_loss.backward(retain_graph=True)
            self.node_optimizer.step()

            # update the edge estimation module
            self.edge_optimizer.zero_grad()
            _, cf_edge_exists, cf_true_edges = self.edge_module(cf_emb_nodes, cf_edge_index[1]) 
            edge_loss = (1-self.alpha) * self.edge_loss_fn(cf_true_edges.double(), cf_edge_exists.double())
            edge_losses.append(edge_loss.item())
            edge_loss.backward()
            self.edge_optimizer.step()
            #######################################################################
            ## Update G network: maximize log(D(G(z)))
            self.prepare_generator_for_training()

            f_node_features, f_edge_index, f_edge_attr, _, f_batch, _ = next(generator_loader)
            y_fake = torch.ones((len(torch.unique(f_batch[1])),)).to(self.device)
            f_embedded_nodes = self.generator(f_node_features[1], f_edge_index[1],
                                              f_edge_attr[1], f_batch[1])
            f_pred = self.discriminator(f_embedded_nodes)
            # calculate the loss
            loss = torch.mean(self.loss_fn(f_pred.double(), y_fake.double()))
            loss.backward()
            G_losses.append(loss.item())
            self.generator_optimizer.step()
                
            self.context.logger.info(f'Epoch {epoch}\t Loss_D = {np.mean(D_losses): .4f}\t Loss_G = {np.mean(G_losses): .4f}\t rec_loss = {np.mean(rec_losses)}\t edge_loss = {np.mean(edge_losses)}')
  
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
        else:
            in_embed_dim, out_embed_dim, num_translator_layers = 10, 4, 4

        
        #Check if the generator exist or build with its defaults:
        init_dflts_to_of(self.local_config, 
                         'generator',
                         dflt_generator,
                         k=self.dataset.num_nodes,
                         in_embed_dim=in_embed_dim,
                         out_embed_dim=out_embed_dim,
                         num_translator_layers=num_translator_layers,
                         node_features=self.dataset.num_node_features())
        #Check if the discriminator exist or build with its defaults:
        init_dflts_to_of(self.local_config,
                         'discriminator',
                         dflt_discriminator,
                         num_nodes=self.dataset.num_nodes,
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
                         num_nodes=self.dataset.num_nodes,
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



    def take_oracle_predictions(self, instances, y_true):
        oracle_scores = [self.oracle.predict_proba(inst)[1-self.explainee_label] for inst in instances]
        oracle_scores = np.array(oracle_scores, dtype=float).squeeze()
        oracle_scores = torch.tensor(oracle_scores, dtype=torch.float).to(self.device)
        return oracle_scores