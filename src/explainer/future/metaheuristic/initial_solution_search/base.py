from typing import Generator
from src.dataset.instances.graph import GraphInstance


class InitialSolutionSearcher:
    def __init__(self):
        pass
        
    def search(self, graph: GraphInstance, tags: list[(int, int)]) -> Generator[set[int], None, None]:
        pass