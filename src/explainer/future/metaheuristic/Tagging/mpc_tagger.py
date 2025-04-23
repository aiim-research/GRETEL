from src.dataset.instances.graph import GraphInstance
from src.explainer.future.metaheuristic.Tagging.base import Tagger
import numpy as np
import random

class MPCTagger(Tagger):
    
    def __init__(self):
        super().__init__()
        
    def tag(self, graph: GraphInstance) -> list[(int, int)]:
        self.G = graph
        self.N = graph.num_nodes
        self.EPlus = int((self.N * (self.N-1)) / 2)
        
        INF = 1e18
        
        dist = np.zeros((self.G.num_nodes, self.G.num_nodes))
        freq = np.zeros((self.G.num_nodes, self.G.num_nodes))
        
        for i in range(self.G.num_nodes):
            for j in range(self.G.num_nodes):
                if i == j:
                    dist[i, j] = 0
                else:
                    dist[i, j] = self.G.data[i, j] if self.G.data[i, j] > 0 else INF
        
        for i in range(self.G.num_nodes):
            for j in range(self.G.num_nodes):
                for k in range(self.G.num_nodes):
                    dist[j, k] = min(dist[j, k], dist[j, i] + dist[i, k])
        
        for i in range(self.G.num_nodes):
            for j in range(self.G.num_nodes):
                for u in range(self.G.num_nodes):
                    for v in range(self.G.num_nodes):
                        if dist[u, v] == dist[u, i] + self.G.data[i, j] + dist[j, v]:
                            freq[i, j] += 1
        
        edges_with_freq = []
        for i in range(self.G.num_nodes - 1):
            for j in range(i + 1, self.G.num_nodes):
                edges_with_freq.append(((i, j), freq[i, j] + freq[j, i]))
        
        edges_with_freq = sorted(edges_with_freq, key=lambda x: x[1], reverse=True)
        result = [edge for edge, _ in edges_with_freq]
        return result
    
    def swap(self, solution : set[int], i: int):  
        self.add(solution, i)
        self.remove(solution, i)
        
        return solution
    
    def add(self, solution : set[int], i : int):
        # asumming solution have the same order that edges_with_frequency
        # asumming that is 1-indexed and EPLUS is not included (view simple tagger)
        universe: set[int] = set()
        it = 1

        while it < self.EPlus and len(universe) < 2*i:
            if it in universe:
                it += 1  # Ensure 'it' increments to avoid infinite loop
                continue
            universe.add(it)
            it += 1  # Increment 'it' after adding to universe

        
        if len(universe) < i:
            raise ValueError("Not enough available numbers to add.")
        
        numbers_to_add = random.sample(universe, i)
        
        solution.update(numbers_to_add)
        
        return solution
    
    def remove(self, solution : set[int], i: int):
        numbers_to_remove = random.sample(solution, i)
        
        solution.difference_update(numbers_to_remove)
        
        return solution