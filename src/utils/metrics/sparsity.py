from src.dataset.instances.graph import GraphInstance
from src.utils.metrics.ged import graph_edit_distance_metric


def sparsity_metric(instance_1: GraphInstance, instance_2: GraphInstance) -> float:
    divided = number_of_structural_features(instance_1)
    if divided == 0:
        return 0
    matrix_1 = instance_1.data
    matrix_2 = instance_2.data
    directed = instance_1.directed and instance_2.directed
    return graph_edit_distance_metric(matrix_1, matrix_2, directed) / divided

def number_of_structural_features(instance: GraphInstance) -> float:
    nx_repr = instance.get_nx()
    divided = 1 if instance.directed else 2
    return len(nx_repr.edges) / divided + len(nx_repr.nodes)
