import networkx as nx
from src.dataset.instances.graph import GraphInstance
from src.explainer.future.metaheuristic.Tagging.base import Tagger
import numpy as np
import random

class ColoringTagger(Tagger):
    
    def __init__(self):
        super().__init__()
    
    
    def tag(self, graph: GraphInstance) -> list[(int, int)]:
        self.G = graph.get_nx()
        self.N = graph.num_nodes
        self.EPlus = int((self.N * (self.N-1)) / 2)
        
        edges = list(self.G.edges())
        m = len(edges)
        H = nx.Graph()
        H.add_nodes_from(range(m))
        
        for i in range(m):
            for j in range(i+1, m):
                if (edges[i][0] == edges[j][0] or edges[i][0] == edges[j][1] or 
                    edges[i][1] == edges[j][0] or edges[i][1] == edges[j][1]):
                    H.add_edge(i, j)
                    
        colors = nx.greedy_color(H)
        edges_with_colors = [(edges[i], colors[i]) for i in range(m)]
    
        edges_with_colors.sort(key=lambda x: x[1])
        
        for u in range(self.N-1):
            for v in range(u+1, self.N):
                if graph.data[u, v] == 0 or graph.data[v, u] == 0:
                    edges_with_colors.append(((u, v), -1))
        
        return [edge for edge, _ in edges_with_colors]
    
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