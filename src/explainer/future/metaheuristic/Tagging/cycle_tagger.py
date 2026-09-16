import networkx as nx
from src.dataset.instances.graph import GraphInstance
from src.explainer.future.metaheuristic.Tagging.base import Tagger
import numpy as np
import random

class CycleTagger(Tagger):
    
    def __init__(self):
        super().__init__()
    
    def tag(self, graph: GraphInstance) -> list[(int, int)]:
        self.G = graph
        self.N = graph.num_nodes
        self.EPlus = int((self.N * (self.N-1)) / 2)
        nx_graph = self.G.get_nx()  
        
        edge_cycle_count = {edge: 0 for edge in nx_graph.edges()}
        
        cycles = list(nx.cycle_basis(nx_graph.to_undirected()))
        
        for cycle in cycles:
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)] 
                if (u, v) in edge_cycle_count:
                    edge_cycle_count[(u, v)] += 1
                elif (v, u) in edge_cycle_count: 
                    edge_cycle_count[(v, u)] += 1
        
        edges_with_cycles = [(edge, count) for edge, count in edge_cycle_count.items()]
        
        edges_with_cycles.sort(key=lambda x: x[1], reverse=True)
        
        for u in range(self.N-1):
            for v in range(u+1, self.N):
                if graph.data[u, v] == 0 and graph.data[v, u] == 0:
                    edges_with_cycles.append(((u, v), -1))
        
        return [edge for edge, _ in edges_with_cycles]
    
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