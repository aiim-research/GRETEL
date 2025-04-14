from src.dataset.instances.graph import GraphInstance
from src.explainer.future.metaheuristic.Tagging.base import Tagger
import numpy as np

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
                        if dist[u, v] == dist[u, i] + dist[i, j] + dist[j, v]:
                            freq[i, j] += 1
        
        edges_with_freq = []
        for i in range(self.G.num_nodes - 1):
            for j in range(i + 1, self.G.num_nodes):
                edges_with_freq.append(((i, j), freq[i, j]))
                if self.G.directed:
                    edges_with_freq.append(((j, i), freq[j, i]))
        
        edges_with_freq = sorted(edges_with_freq, key=lambda x: x[1], reverse=True)
        self.edges_with_freq = edges_with_freq
        result = [edge for edge, _ in edges_with_freq]
        return result
    
    def swap(self, solution : set[int], i: int):  
        self.remove_random(solution, i)
        self.add_random(solution, i)
        
        return solution
    
    def add(self, solution : set[int], i : int):
        # asumming solution have the same order that edges_with_frequency
        # asumming that is 1-indexed and EPLUS is not included (view simple tagger)
        it = 1
        universe : set[int] = {}
        while it < self.EPlus and universe.size() < 2*i:
            if it in universe:
                continue
            universe.add(it)
        
        if len(universe) < i:
            raise ValueError("Not enough available numbers to add.")
        
        numbers_to_add = random.sample(universe, i)
        
        solution.update(numbers_to_add)
        
        return solution
    
    def remove(self, solution : set[int], i: int):
        numbers_to_remove = random.sample(solution, i)
        
        solution.difference_update(numbers_to_remove)
        
        return solution