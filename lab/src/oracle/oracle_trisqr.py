from src.core.oracle_base import Oracle 

import numpy as np
import networkx as nx

class TrianglesSquaresOracle(Oracle):

    def init(self):
        super().init()
        self.model = ""

    def real_fit(self):
        pass


    def _real_predict(self, data_instance):

        if self.is_triangle(data_instance):
            return 0
        else:
            return 1
        
        
    def _real_predict_proba(self, data_instance):
        # softmax-style probability predictions
        if self.is_triangle(data_instance):
            return np.array([1,0])
        else:
            return np.array([0,1])
        
        
    def is_triangle(self, data_instance):
        g = data_instance.get_nx()

        num_nodes = len(g.nodes)
        num_edges = len(g.edges)

        # Check for triangle (3 nodes, 3 edges, all connected)
        if num_nodes == 3 and num_edges == 3:
            return True
        
        # Check for triangle with isolated node (4 nodes, 3 edges)
        elif num_nodes == 4 and num_edges == 3:
            # Find isolated node
            isolated_nodes = list(nx.isolates(g))
            if len(isolated_nodes) == 1:
                return True

        # Check for square (4 nodes, 4 edges, all connected)
        elif num_nodes == 4 and num_edges == 4:
            # Check if it's a cycle of length 4
            if nx.cycle_basis(g) and len(nx.cycle_basis(g)[0]) == 4:
                return False
        
        return False
        
    