
from src.dataset.instances.graph import GraphInstance


class Tagger:
    def __init__(self):
        pass
        
    def tag(self, graph: GraphInstance) -> list[(int, int)]:
        pass
    
    def get_indices(self, labels: list[(int, int)], tuples: list[(int, int)]) -> set[int]:
        result = set()
        for (i, j) in tuples:
            if (i, j) in labels:
                result.add(labels.index((i, j)))
        return sorted(result)