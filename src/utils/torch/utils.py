import torch

def rebuild_adj_matrix(num_nodes: int, edge_indices, edge_features,device="cpu"):    
    truth = torch.zeros(size=(num_nodes, num_nodes)).double().to(device)
    truth[edge_indices[0,:], edge_indices[1,:]] = edge_features
    return truth