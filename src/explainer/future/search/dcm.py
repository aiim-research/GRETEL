import numpy as np
from src.core.explainer_base import Explainer
from src.core.trainable_base import Trainable
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.dataset.instances.graph import GraphInstance
from src.utils.metrics.ged import GraphEditDistanceMetric

class DCM(Explainer, Trainable):
   
    def init(self):
        self.device = "cpu"
        self.distance_metric = GraphEditDistanceMetric()
        super().init()
    
    def real_fit(self):
        super().real_fit()

    def fit(self):
        # Get the category of the graphs
        categorized_graph = [(self.oracle.predict(graph), graph) for graph in self.dataset.instances]
        
        # Groups the graph by category
        graphs_by_category = {}
        for category, graph in categorized_graph:
            if category not in graphs_by_category:
                graphs_by_category[category] = []
            graphs_by_category[category].append(graph)
        
        # Get the medoid of each category
        medoids = {}
        for category, graphs in graphs_by_category.items():
            graphs_distance_total = []
            
            for graph in graphs:
                distance = 0
                
                for category_, graphs_ in graphs_by_category.items():
                    if category == category_:
                        continue
                    for graph_ in graphs_: 
                        distance += self.distance_metric.evaluate(graph, graph_)
                
                graphs_distance_total.append((graph, distance))
            
            min_distance = float('inf')
            medoid = None
            
            for graph, distance in graphs_distance_total:
                if min_distance > distance:
                    min_distance = distance
                    medoid = graph
            
            medoids[category] = medoid
        
        self.model = medoids
        super().fit()
    
    def explain(self, instance):
        # Get the category of the instance
        category = self.oracle.predict(instance)
        
        # Get the closest medoid to the instance that belong to a different category 
        min_distance = float('inf')
        closest_medoid = None
       
        for other_category, medoid in self.model.items():
            if other_category != category:
                distance = self.distance_metric.evaluate(instance, medoid)
                if distance < min_distance:
                    min_distance = distance
                    closest_medoid = medoid       

        print(closest_medoid)
        # Create a graph's instance of the closest medoid
        cf_instance = GraphInstance(id=closest_medoid.id, label=closest_medoid.label, data=closest_medoid.data, node_features=closest_medoid.node_features)

        exp = LocalGraphCounterfactualExplanation(context=self.context, dataset=self.dataset, oracle=self.oracle, explainer=self, input_instance=instance, counterfactual_instances=[cf_instance])

        return exp