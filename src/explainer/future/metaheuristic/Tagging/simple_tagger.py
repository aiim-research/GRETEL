from src.dataset.instances.graph import GraphInstance
from src.explainer.future.metaheuristic.Tagging.base import Tagger

class SimpleTagger(Tagger):
    
    def __init__(self):
        super().__init__()
        
    def tag(self, graph: GraphInstance) -> list[(int, int)]:
        self.G = graph
        
        result = []
        for i in range(self.G.num_nodes - 1):
            for j in range(i + 1, self.G.num_nodes):
                result.append((i,j))
                if(self.G.directed):
                    result.append((j,i))
        return result