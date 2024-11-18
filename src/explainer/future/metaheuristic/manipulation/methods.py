import numpy as np
from scipy.linalg import expm

def average_smoothing(data, edge_features, iterations=1):
    smoothed_features = edge_features.copy()
    num_nodes = edge_features.shape[0]
    for _ in range(iterations):
        new_features = np.zeros_like(smoothed_features)
        for i in range(num_nodes):
            neighbors = np.where(data[i] > 0)[0]  # Find neighbors of node i
            if len(neighbors) > 0:
                # Take the average of the neighbor features
                new_features[i] = np.mean(smoothed_features[neighbors], axis=0)
            else:
                # If no neighbors, keep the feature as is
                new_features[i] = smoothed_features[i]
        smoothed_features = new_features
    return smoothed_features

def average_smoothing_zero(data, edge_features, iterations=1):
    smoothed_features = edge_features.copy()
    num_nodes = edge_features.shape[0]
    for _ in range(iterations):
        new_features = np.zeros_like(smoothed_features)
        for i in range(num_nodes):
            neighbors = np.where(data[i] > 0)[0]  # Find neighbors of node i
            if len(neighbors) > 0:
                # Take the average of the neighbor features
                new_features[i] = np.mean(smoothed_features[neighbors], axis=0)
            else:
                # If no neighbors, keep the feature as 0
                new_features[i] = 0
        smoothed_features = new_features
    return smoothed_features

def weighted_smoothing(data, edge_features, iterations=1):
    smoothed_features = edge_features.copy()
    num_nodes = edge_features.shape[0]
    for _ in range(iterations):
        new_features = np.zeros_like(smoothed_features)
        for i in range(num_nodes):
            neighbors = np.where(data[i] > 0)[0]  # Find neighbors of node i
            if len(neighbors) > 0:
                degrees = np.sum(data[neighbors], axis=1)  # Degree of neighbors
                weights = degrees / np.sum(degrees)  # Normalize weights
                new_features[i] = np.sum(smoothed_features[neighbors] * weights[:, None], axis=0)
            else:
                new_features[i] = smoothed_features[i]
        smoothed_features = new_features
    return smoothed_features

def laplacian_regularization(data, edge_features, lambda_reg=0.01, iterations=1):
    D = np.diag(np.sum(data, axis=1))
    # Graph Laplacian
    L = D - data
    F = edge_features.copy()

    for _ in range(iterations):
        F = (1 - lambda_reg) * F - lambda_reg * np.dot(L, F)
    
    return F
    
def feature_aggregation(data, edge_features, alpha=0.5, iterations=1):
    aggregated_features = edge_features.copy()
    num_nodes = edge_features.shape[0]
    for _ in range(iterations):
        new_features = np.zeros_like(aggregated_features)
        for i in range(num_nodes):
            neighbors = np.where(data[i] > 0)[0]  # Find neighbors of node i
            if len(neighbors) > 0:
                neighbor_mean = np.mean(aggregated_features[neighbors], axis=0)
                new_features[i] = alpha * aggregated_features[i] + (1 - alpha) * neighbor_mean
            else:
                new_features[i] = aggregated_features[i]
        aggregated_features = new_features
    return aggregated_features

def heat_kernel_diffusion(data, edge_features, t=0.5):
    # Degree matrix and graph Laplacian
    D = np.diag(np.sum(data, axis=1))
    L = D - data
    
    H_t = expm(-t * L)
    
    diffused_features = np.dot(H_t, edge_features)
    return diffused_features

def random_walk_diffusion(data, edge_features, steps=1):
    degrees = np.sum(data, axis=1)
    num_nodes = edge_features.shape[0]
    D_inv = np.zeros_like(data, dtype=float)
    for i in range(num_nodes):
        if degrees[i] > 0:
            D_inv[i, i] = 1.0 / degrees[i]
        else:
            D_inv[i, i] = 0.0  # Handle zero-degree nodes safely

    # Transition matrix
    P = np.dot(D_inv, data)

    # Diffuse features through random walk
    diffused_features = edge_features.copy()
    for _ in range(steps):
        diffused_features = np.dot(P, diffused_features)

    return diffused_features