from src.dataset.data_instance_features import DataInstanceWFeatures
from typing import List

class CausalDataInstance(DataInstanceWFeatures):
    
    
    def __init__(self,
                 id=None,
                 name: str = None,
                 graph=None,
                 graph_dgl=None,
                 graph_label: int = None,
                 node_labels: dict = None,
                 edge_labels: dict = None,
                 mcd: int = None,
                 features: List[float] = None,
                 causality: List[float] = None) -> None:
        self._causality = causality
        super().__init__(id, name, graph,
                         graph_dgl, graph_label,
                         node_labels, edge_labels, mcd, features)
    
    @property
    def causality(self):
        return self._causality
    
    @causality.setter
    def causality(self, newcausality):
        self._causality = newcausality