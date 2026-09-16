from src.dataset.instances.graph import GraphInstance
import numpy as np
import random as rd

class FeatureExtractor:
    def __init__(self, graph: GraphInstance, oracle_prediction):
        self.graph = graph
        self.oracle_prediction = oracle_prediction
        self.extract_features()

    def extract_features(self):
        self.max_degree = self.max_degree_feature()
        self.min_degree = self.min_degree_feature()
        self.conex_components = self.conex_components_feature()
        self.density = self.density_feature()
        self.mean_degree = self.mean_degree_feature()
        self.variance_degree = self.variance_degree_feature()
        self.histogram_degree = self.histogram_degree_feature()
        self.wedge_count = self.wedge_count_feature()
        self.triangle_count = self.triangle_count_feature()
        self.approximate_diameter = self.approximate_diameter_feature()
        self.degree_asortativity = self.degree_asortativity_feature()
        self.is_ciclic = self.is_ciclic_feature()
        self.is_directed = self.graph.directed
        self.n = self.graph.num_nodes
        self.m = self.graph.num_edges

    def max_degree_feature(self):
        vertex = []
        DG = 0
        for i in self.graph.nodes():
            if self.graph.degree(i) == DG:
                vertex.append(i)
            if self.graph.degree(i) > DG:
                DG = self.graph.degree(i)
                vertex = [i]
        return DG, vertex
    
    def min_degree_feature(self):
        dG = np.inf
        vertex = []
        for i in self.graph.nodes():
            if self.graph.degree(i) == dG:
                vertex.append(i)
            if self.graph.degree(i) < dG:
                dG = self.graph.degree(i)
                vertex = [i]
        return dG, vertex
    
    def conex_components_feature(self):
        visited = set()
        components = []

        def dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited:
                    dfs(neighbor, component)

        for vertex in self.graph.nodes():
            if vertex not in visited:
                component = []
                dfs(vertex, component)
                components.append(component)

        return components
    
    def density_feature(self):
        n = self.graph.num_nodes
        m = self.graph.num_edges
        if n <= 1:
            return 0
        if self.graph.is_directed:
            density = m / (n * (n - 1))
        else:
            density = (2 * m) / (n * (n - 1))
        return density
    
    def mean_degree_feature(self):
        total_degree = sum(self.graph.degree(node) for node in self.graph.nodes())
        mean_degree = total_degree / self.graph.num_nodes if self.graph.num_nodes > 0 else 0
        return mean_degree
    
    def variance_degree_feature(self):
        mean_degree = self.mean_degree_feature()
        variance = sum((self.graph.degree(node) - mean_degree) ** 2 for node in self.graph.nodes()) / self.graph.num_nodes if self.graph.num_nodes > 0 else 0
        return variance
    
    def histogram_degree_feature(self, bins=10):
        degrees = [self.graph.degree(node) for node in self.graph.nodes()]
        histogram, bin_edges = np.histogram(degrees, bins=bins, range=(0, max(degrees) if degrees else 1))
        return histogram, bin_edges
    
    def wedge_count_feature(self):
        wedge_count = 0
        for node in self.graph.nodes():
            deg = self.graph.degree(node)
            if deg >= 2:
                wedge_count += (deg * (deg - 1)) // 2
        return wedge_count
    
    def triangle_count_feature(self):
        triangle_count = 0
        for node in self.graph.nodes():
            neighbors = self.graph.neighbors(node)
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[j] in self.graph.neighbors(neighbors[i]):
                        triangle_count += 1
        return triangle_count // 3  # Each triangle is counted three times
    
    def approximate_diameter_feature(self):
        def bfs(start_node):
            visited = {start_node}
            queue = [(start_node, 0)]
            max_distance = 0

            while queue:
                current_node, distance = queue.pop(0)
                max_distance = max(max_distance, distance)

                for neighbor in self.graph.neighbors(current_node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))

            return max_distance

        max_diameter = 0
        for node in rd.choices (self.graph.nodes(), k=min(10, self.graph.num_nodes)):
            max_diameter = max(max_diameter, bfs(node))

        return max_diameter
    
    def degree_asortativity_feature(self):
        edges = []
        for node in self.graph.nodes():
            for neighbor in self.graph.neighbors(node):
                if node < neighbor:  # To avoid double counting
                    edges.append((self.graph.degree(node), self.graph.degree(neighbor)))

        if not edges:
            return 0

        degrees_u, degrees_v = zip(*edges)
        mean_u = np.mean(degrees_u)
        mean_v = np.mean(degrees_v)
        std_u = np.std(degrees_u)
        std_v = np.std(degrees_v)

        if std_u == 0 or std_v == 0:
            return 0

        covariance = np.mean([(u - mean_u) * (v - mean_v) for u, v in edges])
        assortativity = covariance / (std_u * std_v)

        return assortativity
    
    def is_ciclic_feature(self):
        visited = set()
       

        def is_cyclic_util(v, parent):
            visited.add(v)

            for neighbor in self.graph.neighbors(v):
                if neighbor in visited and neighbor != parent: 
                    return True
                if neighbor not in visited:
                    if is_cyclic_util(neighbor, v):
                        return True
            return False

        for node in self.graph.nodes():
            if node not in visited:
                if is_cyclic_util(node, -1):
                    return True
        return False
    






    

