import time
import torch
import torch.optim as optim
import numpy as np
from torch_geometric.data import Data
from typing import Optional
from omegaconf import DictConfig
from .graph_perturber import GraphPerturber
from torch_geometric.data import Data

from src.core.explainer_base import Explainer
from src.dataset.instances.graph import GraphInstance
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.dataset.utils.dataset_torch import TorchGeometricDataset


class COMBINEX(Explainer):
    """
    Combined Explainer class that returns counterfactual subgraph by combining
    edge and node perturbations with a dynamic scheduler.
    """
    
    def init(self):
        """Initialize the explainer with device and configuration."""
        super().init()
        
        # Set device
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        
        # Initialize best loss tracking
        self.best_loss = np.inf
        
        # Get configuration parameters with defaults
        self.num_epochs = self.local_config['parameters'].get('optimizer', {}).get('num_epochs', 100)
        self.timeout = self.local_config['parameters'].get('timeout', 3600)  # Default 1 hour
        
        # Set reproducibility if seed is provided
        seed = self.local_config['parameters'].get('seed', None)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    
    def explain(self, instance: GraphInstance):
        """
        Explain an instance by finding a counterfactual graph.
        
        Args:
            instance: GraphInstance to explain
            
        Returns:
            LocalGraphCounterfactualExplanation containing the counterfactual instance(s)
        """
        # Ensure instance has dataset reference
        if instance._dataset is None:
            instance._dataset = self.dataset
        
        # Convert GraphInstance to torch_geometric Data
        graph_data = self._graph_instance_to_data(instance)
        # Ensure graph_data has batch attribute
        """if not hasattr(graph_data, 'batch') or graph_data.batch is None:
            graph_data.batch = torch.zeros(graph_data.x.shape[0], dtype=torch.long, device=self.device)
        else:
            graph_data.batch = graph_data.batch.to(self.device)"""
        
        self.graph_perturber = GraphPerturber(
            cfg=self.local_config,
            model=self.oracle,
            datainfo=self.dataset,
            graph=graph_data,
            device=self.device
        ).to(self.device)
                
        # Get optimizer
        self.optimizer = self.get_optimizer(self.local_config, self.graph_perturber)
        
        best_cf_example = None
        start = time.time()
        
        # Training loop
        for epoch in range(self.num_epochs):
            # Check timeout
            if time.time() - start > self.timeout:
                break
            
            new_sample = self._train(graph_data, instance, epoch)
            
            if new_sample is not None:
                best_cf_example = new_sample
        
        if best_cf_example is not None:
            cf_instance = self._data_to_graph_instance(best_cf_example, instance)
            return cf_instance        
        
        return instance
            
    def _train(self, graph: Data, original_instance: GraphInstance, epoch: int) -> Optional[Data]:
        """
        Trains the graph perturber for one epoch and returns a counterfactual example if found.
        
        Args:
            graph: The input graph data (torch_geometric Data).
            original_instance: The original GraphInstance for reference.
            epoch: The current epoch number.
        
        Returns:
            Data: The counterfactual example if found, otherwise None.
        """
        self.optimizer.zero_grad()
        
        """# Ensure batch is available
        if not hasattr(graph, 'batch') or graph.batch is None:
            batch = torch.zeros(graph.x.shape[0], dtype=torch.long, device=self.device)
        else:
            batch = graph.batch.to(self.device)"""
        
        # Forward pass through graph perturber
        differentiable_output = self.graph_perturber.forward(graph.x.to(self.device), graph.batch)
        model_out, V_pert, EP_x = self.graph_perturber.forward_prediction(graph.x.to(self.device), graph.batch)
        
        # Get predictions
        y_pred_new_actual = torch.argmax(model_out, dim=1)
        y_pred_differentiable = torch.argmax(differentiable_output, dim=1)
        
        # Calculate losses
        edge_loss, cf_edges = self.graph_perturber.edge_loss(graph)
        node_loss, _ = self.graph_perturber.node_loss(graph)
        
        # Get alpha scheduler value
        alpha = self._get_alpha(epoch, edge_loss, node_loss)
        
        # Calculate eta: 1 if prediction changed, 0 otherwise
        original_label = torch.tensor([self.oracle.predict(original_instance)], device=self.device, dtype=torch.long)
        eta = ((y_pred_new_actual != original_label) | (original_label != y_pred_differentiable)).float()
        
        # Calculate prediction loss
        # Note: For counterfactuals, we might want to maximize loss to encourage label flip
        # Adjust this based on your specific loss formulation
        loss_pred = -torch.nn.functional.cross_entropy(
            differentiable_output, 
            original_label
        )
        
        # Combined loss
        loss = eta * loss_pred + (1 - alpha) * edge_loss + alpha * node_loss
        
        # Backward pass
        loss.backward()
        self.optimizer.step()

        counterfactual = None
        # Check if we found a valid counterfactual (prediction should be different from original)
        # Note: Original code checked for equality, but counterfactuals typically require different predictions
        # Adjust this condition based on your specific requirements
        if y_pred_new_actual != original_label and loss.item() < self.best_loss:
            print("Building counterfactual graph")
            counterfactual = self._build_counterfactual_graph_gc(
                x=V_pert,
                edge_index=cf_edges,
                graph=graph,
                oracle=self.oracle,
                output_actual=model_out,
                device=self.device
            )
            
            self.best_loss = loss.item()
        
        return counterfactual
    
    def _get_alpha(self, epoch: int, edge_loss: torch.Tensor, node_loss: torch.Tensor) -> float:
        """
        Scheduler for the alpha value that implements different policies.
        
        Args:
            epoch: The current epoch number.
            edge_loss: Current edge loss value.
            node_loss: Current node loss value.
            
        Returns:
            float: The scheduled alpha value.
        """
        scheduler_config = self.local_config['parameters'].get('scheduler', {})
        policy = scheduler_config.get('policy', 'constant')
        
        if policy == "linear":
            # Linear decay
            alpha = max(0.0, 1.0 - epoch / self.num_epochs)
        elif policy == "exponential":
            # Exponential decay
            decay_rate = scheduler_config.get('decay_rate', 10.0)
            alpha = max(0.0, np.exp(-epoch / decay_rate))
        elif policy == "sinusoidal":
            # Sinusoidal decay
            alpha = max(0.0, 0.5 * (1 + np.cos(np.pi * epoch / self.num_epochs)))
        elif policy == "dynamic":
            # Dynamic adjustment based on loss values
            alpha = 0.0 if edge_loss.item() > node_loss.item() else 1.0
        else:
            # Default to a constant alpha
            alpha = scheduler_config.get('initial_alpha', 0.5)
        
        return alpha
    
    def _graph_instance_to_data(self, instance: GraphInstance) -> Data:
        """
        Convert GraphInstance to torch_geometric Data.
        
        Args:
            instance: GraphInstance to convert
            
        Returns:
            Data: torch_geometric Data object
        """
        return TorchGeometricDataset.to_geometric(instance, label=instance.label)
    
    def _data_to_graph_instance(self, data: Data, original_instance: GraphInstance) -> GraphInstance:
        """
        Convert torch_geometric Data back to GraphInstance.
        
        Args:
            data: torch_geometric Data object
            original_instance: Original GraphInstance for reference (id, dataset, etc.)
            
        Returns:
            GraphInstance: Converted graph instance
        """
        # Convert edge_index back to adjacency matrix
        num_nodes = data.x.shape[0]
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        
        if data.edge_index is not None and data.edge_index.numel() > 0:
            edge_index = data.edge_index.cpu().detach().numpy()
            if edge_index.shape[0] == 2:  # [2, num_edges]
                for i in range(edge_index.shape[1]):
                    row, col = edge_index[0, i], edge_index[1, i]
                    if row < num_nodes and col < num_nodes:
                        weight = data.edge_attr[i].item() if data.edge_attr is not None and i < len(data.edge_attr) else 1.0
                        adj_matrix[row, col] = weight
        
        # Get node features
        node_features = data.x.cpu().detach().numpy().astype(np.float32)
        
        # Get edge features and weights
        edges = np.nonzero(adj_matrix)
        num_edges = len(edges[0])
        
        if num_edges > 0:
            edge_features = np.ones((num_edges, 1), dtype=np.float32)
            if data.edge_attr is not None and len(data.edge_attr) > 0:
                edge_weights = data.edge_attr.cpu().detach().numpy().flatten().astype(np.float32)
                if len(edge_weights) != num_edges:
                    # If mismatch, use adjacency matrix values
                    edge_weights = adj_matrix[edges].astype(np.float32)
            else:
                edge_weights = adj_matrix[edges].astype(np.float32)
        else:
            edge_features = np.array([], dtype=np.float32).reshape(0, 1)
            edge_weights = np.array([], dtype=np.float32)
        
        # Create new GraphInstance
        cf_instance = GraphInstance(
            id=original_instance.id,  # Keep same ID or modify as needed
            label=data.y.item() if data.y is not None else original_instance.label,
            data=adj_matrix,
            node_features=node_features,
            edge_features=edge_features,
            edge_weights=edge_weights,
            graph_features=original_instance.graph_features,
            dataset=original_instance._dataset,
            directed=original_instance.directed
        )
        
        return cf_instance
    
    def get_optimizer(self, cfg, model):
        """
        Create an optimizer based on configuration.
        
        Args:
            cfg: Configuration (DictConfig or dict)
            model: Model to optimize
            
        Returns:
            Optimizer instance
        """
        # Handle both DictConfig and regular dict
        if isinstance(cfg, DictConfig):
            opt_config = cfg.optimizer
            opt_name = opt_config.name
            opt_lr = opt_config.lr
            opt_momentum = getattr(opt_config, 'n_momentum', 0.0)
        else:
            # GRETEL dict structure
            opt_dict = cfg.get('parameters', {}).get('optimizer', {})
            opt_name = opt_dict.get('name', 'adam')
            opt_lr = opt_dict.get('lr', 0.01)
            opt_momentum = opt_dict.get('n_momentum', 0.0)
        
        if opt_name == "sgd" and opt_momentum == 0.0:
            return optim.SGD(model.parameters(), lr=opt_lr)
        elif opt_name == "sgd" and opt_momentum != 0.0:
            return optim.SGD(model.parameters(), lr=opt_lr, nesterov=True, momentum=opt_momentum)
        elif opt_name == "adadelta":
            return optim.Adadelta(model.parameters(), lr=opt_lr)
        elif opt_name == "adam":
            return optim.Adam(model.parameters(), lr=opt_lr)
        else:
            raise ValueError(f"Optimizer {opt_name} does not exist!")

    def _build_counterfactual_graph_gc(self, x, edge_index, graph, oracle, output_actual, device):
        """
        Build a counterfactual graph using the graph perturber.
        """
        # Get embedding representation if the oracle supports it
        x_projection = None
        if hasattr(oracle, 'get_embedding_repr'):
            try:
                x_projection = torch.mean(oracle.get_embedding_repr(x, edge_index, batch), dim=0)
            except Exception as e:
                # Fallback if get_embedding_repr fails
                # Use mean of node features as projection
                x_projection = torch.mean(x, dim=0)
        else:
            # Fallback: use mean of node features as projection
            x_projection = torch.mean(x, dim=0)
        
        counterfactual = Data(
            x=x.to(device),
            edge_index=edge_index.to(device),
            y=torch.argmax(output_actual, dim=1).to(device),
            x_projection=x_projection.to(device)
        )
        
        return counterfactual

    def check_configuration(self):
        """Check and validate configuration parameters."""
        super().check_configuration()
        
        # Set defaults for optimizer
        if 'optimizer' not in self.local_config['parameters']:
            self.local_config['parameters']['optimizer'] = {}
        if 'num_epochs' not in self.local_config['parameters']['optimizer']:
            self.local_config['parameters']['optimizer']['num_epochs'] = 100
        
        # Set defaults for scheduler
        if 'scheduler' not in self.local_config['parameters']:
            self.local_config['parameters']['scheduler'] = {}
        if 'policy' not in self.local_config['parameters']['scheduler']:
            self.local_config['parameters']['scheduler']['policy'] = 'constant'
        if 'initial_alpha' not in self.local_config['parameters']['scheduler']:
            self.local_config['parameters']['scheduler']['initial_alpha'] = 0.5

