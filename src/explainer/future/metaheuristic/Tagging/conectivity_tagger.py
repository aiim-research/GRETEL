from src.dataset.instances.graph import GraphInstance
from src.explainer.future.metaheuristic.Tagging.base import Tagger
import numpy as np
import random

class ConectivityTagger(Tagger):
    
    def __init__(self):
        super().__init__()
        
    def tag(self, graph: GraphInstance) -> list[(int, int)]:
        self.G = graph
        self.N = graph.num_nodes
        self.EPlus = int((self.N * (self.N-1)) / 2)
        edges_with_flow = []
    
        for u in range(self.G.num_nodes-1):
            for v in range(self.G.num_nodes):
                Gp = self.G.copy()
                Gp.data[u][v] = 0
                Gp.data[v][u] = 0
                
                flow_graph = nx.DiGraph()
                for i in range(self.G.num_nodes):
                    for j in range(self.G.num_nodes):
                        if Gp.data[i][j] > 0:
                            flow_graph.add_edge(i, j, capacity=1)
                
                flow_value, _ = nx.maximum_flow(flow_graph, u, v)
                
                edges_with_flow.append(((u, v), flow_value))
        
        edges_with_flow = sorted(edges_with_flow, key=lambda x: x[1])
        
        result = [edge for edge, _ in edges_with_flow]
        return result
    
    
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