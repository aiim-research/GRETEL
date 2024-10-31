import random
from typing import Generator
from src.dataset.instances.graph import GraphInstance
from src.explainer.future.metaheuristic.initial_solution_search.base import InitialSolutionSearcher


class SimpleSearcher(InitialSolutionSearcher):
    def __init__(self):
        super().__init__()
        
        
    def search(self, graph: GraphInstance, tags: list[(int, int)]) -> Generator[set[int], None, None]:
        self.G = graph
        self.Tags = tags
        n = self.G.num_nodes
        while True:
            yield self.random_set((int)((n * (n - 1))/2))
    
    def random_set(self, n) -> set[int]:
        cardinality = random.randint(1, n)
        
        random_numbers = random.sample(range(0, n), cardinality)
        
        return set(random_numbers)