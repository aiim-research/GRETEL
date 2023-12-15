import numpy as np

class GraphCharacteristicsCalculator:
    
    def __init__(self, data):
        self.data = data
        self._graph_measures = {'graph': {}, 'nodes': {}, 'edges': {}}
        
    def diameter(self):
        if 'diameter' not in self._graph_measures['graph']:
            data = self.data + np.identity(self.data.shape[0])
            # Initialize M and diameter
            M, diameter = data, 1
            # Compute the powers of A until all entries are nonzero
            while np.count_nonzero(M == 0) > 0:
                M = np.matmul(M, data)
                diameter += 1
            # Return the diameter
            self._graph_measures['graph']['diameter'] = diameter
            
    def node_degrees(self):
        if 'degrees' not in self._graph_measures['nodes']:
            degrees = [val for _, val in np.sum(self.data, axis=1)]
            self._graph_measures['nodes']['degrees'] = degrees
