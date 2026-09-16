import torch
from torch_geometric.data import Data   
from torch import Tensor
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from typing import Tuple

from src.dataset.instances.graph import GraphInstance
import numpy as np

from .perturber.pertuber import Perturber


class GraphPerturber(Perturber):
    """
    Graph perturbation model that learns to modify node features and edge weights
    to generate counterfactual explanations.
    """

    def __init__(self, 
                 cfg, 
                 model, 
                 graph: Data,
                 datainfo,  # Can be Dataset or a DataInfo-like object
                 device: str = "cuda") -> None:
        
        super().__init__(cfg=cfg, oracle=model)
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.beta = 0.5
        # Initialize edge perturbation parameters
        num_edges = graph.edge_index.shape[1] if graph.edge_index.numel() > 0 else 0
        self.EP_x = Parameter(torch.ones(num_edges, device=self.device))
        self.graph_sample = graph
        
        # Extract dataset characteristics from datainfo (Dataset or dict-like)
        if hasattr(datainfo, 'num_classes'):
            self.num_classes = datainfo.num_classes
        else:
            self.num_classes = cfg.get('parameters', {}).get('num_classes', 2)
        
        self.num_nodes = graph.x.shape[0]
        
        if hasattr(datainfo, 'num_node_features'):
            self.num_features = datainfo.num_node_features()
        elif hasattr(datainfo, 'num_features'):
            self.num_features = datainfo.num_features
        else:
            self.num_features = graph.x.shape[1] if len(graph.x.shape) > 1 else 1
        
        # Get min_range, max_range, and discrete_mask from datainfo or compute from graph
        if hasattr(datainfo, 'min_range'):
            self.min_range = torch.tensor(datainfo.min_range, device=self.device, dtype=torch.float32)
        else:
            # Compute from graph features
            min_val = float(graph.x.min().item()) if graph.x.numel() > 0 else 0.0
            self.min_range = torch.tensor(min_val, device=self.device, dtype=torch.float32)
        
        if hasattr(datainfo, 'max_range'):
            self.max_range = torch.tensor(datainfo.max_range, device=self.device, dtype=torch.float32)
        else:
            # Compute from graph features
            max_val = float(graph.x.max().item()) if graph.x.numel() > 0 else 1.0
            self.max_range = torch.tensor(max_val, device=self.device, dtype=torch.float32)
        
        if hasattr(datainfo, 'discrete_mask'):
            self.discrete_features_mask: Tensor = torch.tensor(datainfo.discrete_mask, device=self.device, dtype=torch.float32)
        else:
            # Default: assume all features are continuous
            self.discrete_features_mask: Tensor = torch.zeros(self.num_features, device=self.device, dtype=torch.float32)
        
        self.continous_features_mask: Tensor = 1 - self.discrete_features_mask
        
        # Explainer characteristics
        self.discrete_features_addition: bool = True
        
        # Model's parameters for node feature perturbations
        self.P_x = Parameter(torch.zeros(self.num_nodes, self.num_features, device=self.device))
        
        # Graph's components
        self.edge_index = graph.edge_index.to(self.device)
        self.x = graph.x.to(self.device)
        
        # Move to device
        self.to(self.device)
        
    def discretize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Discretizes the input tensor based on the following rules:
        - Values less than or equal to 0.5 are set to 0
        - Values greater than 0.5 are set to 1

        Args:
            tensor (torch.Tensor): The input tensor to be discretized.

        Returns:
            torch.Tensor: The discretized tensor.
        """
        discretized_tensor = torch.where(tensor <= 0.5, 0, 1)
        return discretized_tensor.float()
    
    def forward(self, V_x, batch) -> Tensor:       
        """
        Forward pass for the graph perturber.
        This method perturbs the input features and passes them through the model.
        
        Args:
            V_x (Tensor): The input node features.
            batch (Tensor): The batch indices for the nodes.
            
        Returns:
            Tensor: The output of the model after applying the perturbations.
        """
        tanh_discrete_features = torch.tanh(self.P_x)
        perturbation_discrete_rescaling = self.min_range + (self.max_range - self.min_range) * tanh_discrete_features
        perturbed_discrete_features = perturbation_discrete_rescaling + V_x
        discrete_perturbation = self.discrete_features_mask * torch.clamp(perturbed_discrete_features, min=self.min_range, max=self.max_range)
        continuous_perturbation = self.continous_features_mask * torch.clamp((self.P_x + V_x), min=self.min_range, max=self.max_range)
        return self.oracle.model(perturbed_discrete_features, self.edge_index, torch.sigmoid(self.EP_x), batch)
    
    def forward_prediction(self, V_x, batch):
        """
        Forward pass with discretized perturbations for actual prediction.
        
        Args:
            V_x (Tensor): The input node features.
            batch (Tensor): The batch indices for the nodes.
            
        Returns:
            Tuple[Tensor, Tensor, Tensor]: (model_output, perturbed_features, edge_weights)
        """
        discrete_perturbation = self.discrete_features_mask * torch.round(
            self.min_range + (self.max_range - self.min_range) * torch.tanh(self.P_x) + V_x
        )
        discrete_perturbation = torch.clamp(discrete_perturbation, min=self.min_range, max=self.max_range)
        continuous_perturbation = self.continous_features_mask * torch.clamp(
            (self.P_x + V_x), min=self.min_range, max=self.max_range
        )
        V_pert = discrete_perturbation + continuous_perturbation
        EP_x_discrete = self.discretize_tensor(torch.sigmoid(self.EP_x))
        self.edge_index = self.edge_index.long()
        out = self.oracle.model(V_pert, self.edge_index, EP_x_discrete, batch)          
        return out, V_pert, self.EP_x
    
    def edge_loss(self, graph: Data) -> Tuple[Tensor, Tensor]:
        """
        Calculate edge sparsity loss and return counterfactual edge index.
        
        Args:
            graph (Data): The input graph data.
            
        Returns:
            Tuple[Tensor, Tensor]: (sparsity_loss, cf_edge_index)
        """
        # Generate perturbed edge index (with edge weights)
        cf_edge_weights = torch.sigmoid(self.EP_x)  # Learnable edge weights (perturbations)

        # Graph sparsity loss: Penalize large changes in edge weights
        sparsity_loss = torch.sum(torch.abs(cf_edge_weights - 1))  # Penalize deviations from original weights
        cf_edge_weights_discrete = self.discretize_tensor(cf_edge_weights)
        
        # Filter edges based on discrete weights
        if graph.edge_index.numel() > 0 and len(cf_edge_weights_discrete) > 0:
            mask = cf_edge_weights_discrete == 1
            if mask.any():
                cf_edge_index = graph.edge_index[:, mask]
            else:
                # If no edges are selected, return empty edge index with correct shape
                cf_edge_index = torch.empty((2, 0), dtype=graph.edge_index.dtype, device=self.device)
        else:
            # Empty edge index
            cf_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        return sparsity_loss, cf_edge_index

    def node_loss(self, graph: Data) -> Tuple[Tensor, Tensor]:
        """
        Calculate the node loss for a given graph.
        This method computes the loss for discrete and continuous features of the nodes in the graph.
        The discrete feature loss is calculated using L1 loss, and the continuous feature loss is 
        calculated using Mean Squared Error (MSE) loss. The total loss is the sum of these two losses.
        
        Args:
            graph (Data): The input graph data containing node features.
            
        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the total loss and edge index.
        """
        loss_discrete = F.l1_loss(
            graph.x * self.discrete_features_mask, 
            torch.clamp(self.discrete_features_mask * F.tanh(self.P_x) + graph.x, 
                       self.min_range.item(), self.max_range.item())
        )
        loss_continue = F.mse_loss(
            graph.x * self.continous_features_mask, 
            self.continous_features_mask * (self.P_x + graph.x)
        )
        loss_total = loss_discrete + loss_continue
        
        return loss_total, self.edge_index

