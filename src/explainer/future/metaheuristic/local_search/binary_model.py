from src.core.oracle_base import Oracle
from src.dataset.instances.graph import GraphInstance


class BinaryModel:
    def __init__(self, oracle : Oracle, graphInstance : GraphInstance):
        self.Model = oracle
        self.InitialResponse = oracle.predict(graphInstance)
        
    def classify(self, graph: GraphInstance):
        new_response = self.Model.predict(graph)
        return self.InitialResponse != new_response
        