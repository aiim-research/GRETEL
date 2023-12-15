

class ObliviousBidirectionalDistance():
    def distance(self, g_1,g_2):
        '''Returns a particular version of Graph Edit Distance when only edge changes are considered and
        the graphs are already matched
        '''
        return self._tot_edges(abs(g_1-g_2))


    def _tot_edges(self, g):
        '''Returns the total number of edges for undirected graphs
        '''
        return sum([sum(el) for el in g])/2