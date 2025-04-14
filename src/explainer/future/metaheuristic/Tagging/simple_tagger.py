from src.dataset.instances.graph import GraphInstance
from src.explainer.future.metaheuristic.Tagging.base import Tagger

class SimpleTagger(Tagger):
    
    def __init__(self):
        super().__init__()
        
    def tag(self, graph: GraphInstance) -> list[(int, int)]:
        self.G = graph
        self.N = graph.num_nodes
        self.EPlus = int((self.N * (self.N-1)) / 2)
        
        result = []
        for i in range(self.G.num_nodes - 1):
            for j in range(i + 1, self.G.num_nodes):
                result.append((i,j))
                if(self.G.directed):
                    result.append((j,i))
        return result
    
    def swap(self, solution : set[int], i: int):  
        self.remove_random(solution, i)
        self.add_random(solution, i)
        
        return solution
    
    def add(self, solution : set[int], i: int):
        available_numbers = set(range(1, self.EPlus)) - solution
        
        if len(available_numbers) < i:
            raise ValueError("Not enough available numbers to add.")
        
        numbers_to_add = random.sample(available_numbers, i)
        
        solution.update(numbers_to_add)
        
        return solution
    
    def remove(self, solution : set[int], i: int):
        numbers_to_remove = random.sample(solution, i)
        
        solution.difference_update(numbers_to_remove)
        
        return solution