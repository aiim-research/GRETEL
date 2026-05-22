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
        self.logger = self.context.logger
        self.fold_id = self.local_config['parameters']['fold_id']
        self.proportion = self.local_config['parameters'].get('proportion', 1.0)
        super().init()
    
    def real_fit(self):
        indices = self.dataset.get_split_indices(fold_id=self.fold_id)['train']
        
        # Get the category of the graphs
        categorized_graph = []

        for i in indices:
            graph = self.dataset.get_instance(i)
            category = graph.label
            categorized_graph.append((category, graph))

        # Groups the graph by category
        graphs_by_category = {}
        for category, graph in categorized_graph:
            if category not in graphs_by_category:
                graphs_by_category[category] = []
            graphs_by_category[category].append(graph)

        for category in graphs_by_category:
            n_graphs = len(graphs_by_category[category])
            n_selected = int(n_graphs * self.proportion)
            graphs_by_category[category] = graphs_by_category[category][:n_selected]
        
        # Get the medoid of each category

        index = 0
        total = len(graphs_by_category)

        medoids = {}
        distances = {}
        for category, graphs in graphs_by_category.items():
            graphs_distance_total = []

            self.logger.info("Starting category {}/{}".format(index, total))
            
            n = 0
            for graph in graphs:
                n +=1
                self.logger.info("Processing graph {}/{}".format(n, len(graphs)))
                distance = 0
                
                for category_, graphs_ in graphs_by_category.items():
                    if category == category_:
                        continue
                    for graph_ in graphs_: 
                        key = (graph.id, graph_.id)

                        if key in distances:
                            distance += distances[key]
                        else:
                            distances[key] = self.distance_metric.evaluate(graph, graph_)
                            distances[(key[1], key[0])] = distances[key]
                            
                        distance += distances[key]

                graphs_distance_total.append((graph, distance))
            
            min_distance = float('inf')
            medoid = None
            
            n = 0
            for graph, distance in graphs_distance_total:
                n+=1
                self.logger.info("Processing graph distance {}/{}".format(n, len(graphs)))
                if min_distance > distance:
                    min_distance = distance
                    medoid = graph
            
            medoids[category] = medoid

            self.logger.info("Ending category {}/{}".format(index, total))
            index += 1

        self.model = medoids

        super().real_fit()
    
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