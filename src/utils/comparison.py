from typing import List
import numpy as np
import sys

from src.dataset.instances.base import DataInstance
from src.dataset.instances.graph import GraphInstance
from src.utils.utils import pad_adj_matrix

def get_edge_differences(instance: DataInstance, cf_instance: DataInstance):
    # Summing the two adjacency matrices (the metrices need to have the same size) edges that appear only in one of the two instances are the different ones
    edge_freq_matrix = np.add(instance.data, cf_instance.data)
    diff_matrix = np.where(edge_freq_matrix == 1, 1, 0)
    diff_number = np.count_nonzero(diff_matrix)

    if instance.directed:
        filtered_diff_number = int(diff_number)
    else:
        filtered_diff_number = int(diff_number/2)

    return filtered_diff_number, diff_matrix
    

def get_all_edge_differences(instance: DataInstance, cf_instances: List[DataInstance]):
    # Getting the max explanation instance dimension and padding the original instance
    max_dim = max(instance.data.shape[0], max([exp.data.shape[0] for exp in cf_instances]))
    padded_inst_adj_matrix = pad_adj_matrix(instance.data, max_dim)

    padded_instance = GraphInstance(id=instance.id, data=padded_inst_adj_matrix, label=instance.label, directed=instance.directed)
    instance.dataset.manipulate(padded_instance)

    # Creating a matrix with the edges that were changed in any explanation and in how many explanations they were modified
    edge_change_freq_matrix = np.zeros((max_dim, max_dim))
    min_changes = sys.maxsize
    for exp in cf_instances:
        exp_changes_num, exp_changes_mat = get_edge_differences(padded_instance, pad_adj_matrix(exp.data, max_dim))

        # Aggregating the edges that were modified in each explanation
        edge_change_freq_matrix = np.add(edge_change_freq_matrix, exp_changes_mat)

        # Keeping what was the minimum number of changes performed by any explanation
        if exp_changes_num < min_changes:
            min_changes = exp_changes_num

    # Get the positions of the edge_change_freq
    edges = [(row, col) for row, col in zip(*np.where(edge_change_freq_matrix))]

    # If we are working with directed graphs
    if instance.is_directed:
        filtered_edges = edges
    else: # if we are working with undirected graphs
        filtered_edges = []
        for x, y in edges:
            if (y,x) not in filtered_edges:
                filtered_edges.append((x,y))

    return filtered_edges, min_changes, edge_change_freq_matrix 