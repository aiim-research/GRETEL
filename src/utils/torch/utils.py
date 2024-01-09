import torch

from torch import Tensor
from torch_geometric.utils import cumsum, scatter
from torch_geometric.nn.pool.select import SelectTopK

def rebuild_adj_matrix(num_nodes: int, edge_indices, edge_features,device="cpu"):    
    truth = torch.zeros(size=(num_nodes, num_nodes)).double().to(device)
    truth[edge_indices[0,:], edge_indices[1,:]] = edge_features
    return truth

def topk(x: Tensor, k: int, batch: Tensor) -> Tensor:
    batch = batch.type(torch.int64)
    if k is not None:
        num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce='sum')
        x, x_perm = torch.sort(x.view(-1), descending=True)
        batch = batch[x_perm]
        batch, batch_perm = torch.sort(batch, descending=False, stable=True)
        k = num_nodes.new_full((num_nodes.size(0), ), int(k))
        arange = torch.arange(x.size(0), dtype=torch.long, device=x.device)
        ptr = cumsum(num_nodes)
        batched_arange = arange - ptr[batch]
        mask = batched_arange < k[batch]

        return x_perm[batch_perm[mask]]

    raise ValueError("'k'parameter must be specified")