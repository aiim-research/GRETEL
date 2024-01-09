import torch
from torch import Tensor

from torch_geometric.nn.inits import uniform
from torch_geometric.nn.pool.select import Select, SelectOutput
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import softmax

from torch_geometric.nn.pool.select import SelectTopK
from typing import Callable, Optional, Tuple, Union
from torch_geometric.nn.pool.connect import FilterEdges
from torch_geometric.typing import OptTensor

from src.utils.torch.utils import topk


class TopKPooling(torch.nn.Module):
    
    def __init__(self, in_channels: int, k: int = 2, multiplier: float = 1., nonlinearity: Union[str, Callable] = 'tanh'):
        super().__init__()
        
        self.in_channels = in_channels
        self.k = k
        self.multiplier = multiplier
        self.select = SelectTopK(in_channels, k, nonlinearity)
        self.connect = FilterEdges()
        
    def reset_parameters(self):
        self.select.reset_parameters()
        
    def forward(self, x: Tensor, edge_index: Tensor, 
                edge_attr: Optional[Tensor] = None,
                batch: Optional[Tensor] = None,
                attn: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, OptTensor, OptTensor, Tensor, Tensor]:
        
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
            
        attn = x if attn is None else attn
        select_out = self.select(attn, batch)
        
        perm = select_out.node_index
        score = select_out.weight
        assert score is not None
        
        x = x[perm] * score.view(-1,1)
        x = self.multiplier * x if self.multiplier != 1 else x
            
        connect_out = self.connect(select_out, edge_index, edge_attr, batch.type(torch.int64))
        
        return x, connect_out.edge_index, connect_out.edge_attr, connect_out.batch, perm, score
        
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, {self.k}, multiplier={self.multiplier})')
    

class SelectTopK(Select):
    
    def __init__(self, in_channels: int, k: int = 2, act: Union[str, Callable] = 'tanh'):
        super().__init__()

        self.in_channels = in_channels
        self.k = k
        self.act = activation_resolver(act)

        self.weight = torch.nn.Parameter(torch.empty(1, in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)

    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> SelectOutput:
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        x = x.view(-1, 1) if x.dim() == 1 else x
        score = (x * self.weight).sum(dim=-1)
        
        score = self.act(score / self.weight.norm(p=2, dim=-1))
        
        node_index = topk(score, self.k, batch)

        return SelectOutput(
            node_index=node_index,
            num_nodes=x.size(0),
            cluster_index=torch.arange(node_index.size(0), device=x.device),
            num_clusters=node_index.size(0),
            weight=score[node_index],
        )

    def __repr__(self) -> str:
        arg = f'k={self.k}'
        return f'{self.__class__.__name__}({self.in_channels}, {arg})'
