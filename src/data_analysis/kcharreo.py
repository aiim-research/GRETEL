import numpy as np


def get_node_changes(g1, g2):
    """Returns (common nodes, added nodes, removed nodes)"""

    n_g1 = g1.shape[0]
    n_g2 = g2.shape[0]

    common_nodes_num = min(n_g1, n_g2)
    common_nodes = list(range(0, common_nodes_num))

    if n_g1 == n_g2: # Both graphs have the same size
        removed_nodes = []
        added_nodes = []
    elif n_g1 > n_g2: # G1 has more nodes than G2
        removed_nodes = list(range(common_nodes_num, n_g1))
        added_nodes = []
    else: # G2 has more nodes than G1
        removed_nodes = []
        added_nodes = list(range(common_nodes_num, n_g2))

    return common_nodes, added_nodes, removed_nodes


def get_edge_changes(g1, g2, directed=False):
    """Returns (common_edges_list, added_edges_list, removed_edges_list)"""
    g1_A = g1
    g2_A = g2
    n_g1 = g1.shape[0]
    n_g2 = g2.shape[0]
    common_nodes_num = min(n_g1, n_g2)

    common_edges = []
    added_edges = []
    removed_edges = []

    if directed: # If the graphs are directed

        # Check the edges between the nodes common to both graphs
        for i in range(common_nodes_num):
            for j in range(common_nodes_num):
                if g1_A[i,j] == 1 and g2_A[i,j] == 1:
                    common_edges.append((i,j))
                elif g1_A[i,j] == 0 and g2_A[i,j] == 1:
                    added_edges.append((i,j))
                elif g1_A[i,j] == 1 and g2_A[i,j] == 0:
                    removed_edges.append((i,j))

        # If g2 removed nodes from g1 then all the edges from those nodes were removed
        if n_g1 > n_g2:
            for i in range(common_nodes_num, n_g1):
                for j in range(0, i):
                    if g1_A[i,j] == 1:
                        removed_edges.append((i,j))

            for j in range(common_nodes_num, n_g1):
                for i in range(0, j):
                    if g1_A[i,j] == 1:
                        removed_edges.append((i,j))

        # If g2 added nodes then all the edges to those nodes were added
        elif n_g2 > n_g1:
            for i in range(common_nodes_num, n_g2):
                for j in range(0, j):
                    if g2_A[i,j] == 1:
                        added_edges.append((i,j))
            
            for j in range(common_nodes_num, n_g2):
                for i in range(0, j):
                    if g2_A[i,j] == 1:
                        added_edges.append((i,j))

    else: # The graph is undirected, so we only iterate over half the graph

        # Check the edges between the nodes common to both graphs
        for i in range(common_nodes_num):
            for j in range(i, common_nodes_num):
                if g1_A[i,j] == 1 and g2_A[i,j] == 1:
                    common_edges.append((i,j))
                elif g1_A[i,j] == 0 and g2_A[i,j] == 1:
                    added_edges.append((i,j))
                elif g1_A[i,j] == 1 and g2_A[i,j] == 0:
                    removed_edges.append((i,j))

        # If g2 removed nodes from g1 then all the edges from those nodes were removed
        if n_g1 > n_g2:
            for j in range(common_nodes_num, n_g1):
                for i in range(0, j):
                    if g1_A[i,j] == 1:
                        removed_edges.append((i,j))

        # If g2 added nodes then all the edges to those nodes were added
        elif n_g2 > n_g1:
            for j in range(common_nodes_num, n_g2):
                for i in range(0, j):
                    if g2_A[i,j] == 1:
                        added_edges.append((i,j))

    return common_edges, added_edges, removed_edges





    

# Triangle
triangle = np.array([[0, 1, 1],
                     [1, 0, 1],
                     [1, 1, 0]])

square = np.array([[0, 1, 0, 1],
                   [1, 0, 1, 0],
                   [0, 1, 0, 1],
                   [1, 0, 1, 0]])

d_triangle = np.array([[0, 1, 0],
                       [0, 0, 0],
                       [1, 1, 0]])

d_square = np.array([[0, 1, 0, 0],
                     [0, 0, 0, 0],
                     [0, 1, 0, 1],
                     [1, 0, 0, 0]])

print(get_node_changes(triangle, square))

print(get_node_changes(square, triangle))

common_edges, added_edges, removed_edges = get_edge_changes(triangle, square)
print(f'common edges: {common_edges}')
print(f'added edges: {added_edges}')
print(f'removed edges: {removed_edges}')


common_edges, added_edges, removed_edges = get_edge_changes(d_triangle, d_square, directed=True)
print(f'common edges: {common_edges}')
print(f'added edges: {added_edges}')
print(f'removed edges: {removed_edges}')