from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn


class Perturber(nn.Module, metaclass=ABCMeta):
    """
    Base class for graph perturbation oracles.
    """
    
    def __init__(self, cfg, oracle: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.oracle = oracle
        self._oracle_active = True
    
    def deactivate_oracle(self):
        """Deactivate the oracle (e.g., set to eval mode, disable gradients)."""
        self._oracle_active = False
        if hasattr(self.oracle.model, 'eval'):
            self.oracle.eval()
        for param in self.oracle.model.parameters():
            param.requires_grad = False
    
    def activate_oracle(self):
        """Activate the oracle (e.g., set to train mode, enable gradients)."""
        self._oracle_active = True
        if hasattr(self.oracle.model, 'train'):
            self.oracle.model.train()
        for param in self.oracle.model.parameters():
            param.requires_grad = True
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass through the perturber."""
        pass

