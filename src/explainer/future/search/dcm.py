import numpy as np
from src.core.explainer_base import Explainer
from src.core.trainable_base import Trainable
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.dataset.instances.graph import GraphInstance
from src.utils.metrics.ged import GraphEditDistanceMetric

# Aqui se puede considerar explainer pero realmente es solo un generador, duda con que heredo

class DCM(Explainer, Trainable):
    # Actualmente no necesito nada en la configuracion, ni nada en el init personalizado

    def init(self):
        self.device = "cpu"
        self.distance_metric = GraphEditDistanceMetric()
        super().init()
    
    def real_fit(self):
        super().real_fit()

    def fit(self):
        #if self.model:
        #   return self.model
        
        # Realizar el entrenamiento para guardarlo en cache

        medoids = None

        # Categorizar todos los grafos del dataset
        categorized_graph = [(self.oracle.predict(graph), graph) for graph in self.dataset.instances]
        
        # Agrupar los grafos por categoría
        graphs_by_category = {}
        for category, graph in categorized_graph:
            if category not in graphs_by_category:
                graphs_by_category[category] = []
            graphs_by_category[category].append(graph)
        
        # Calcular el medoide de cada categoría
        medoids = {}
        for category, graphs in graphs_by_category.items():
                features = np.array([graph.data for graph in graphs])
                medoid = np.mean(features, axis=0)
                medoids[category] = medoid
        
        # Guardar los medoides
        self.model = medoids
        super().fit()
        #super().write()
    
    def explain(self, instance):
        
        # Obtener la categoria de la instancia
        category = self.oracle.predict(instance)
        

        # Moverse por cada una de las categorias existentes en el dataset diferentes de la categoria actual, y comparar los medoides de cada una de esas categorias con la instancia actual, devolver el mas cercano
        min_distance = float('inf')
        closest_medoid = None
        for other_category, medoid in self.model.items():
            if other_category != category:
                distance = self.distance_metric.evaluate(instance, medoid)
                if distance < min_distance:
                    min_distance = distance
                    closest_medoid = medoid       

        # Crear una instancia de grafo con el medoid mas cercano
        cf_instance = GraphInstance(id=instance.id, label=instance.label, data=closest_medoid, node_features=instance.node_features)

        # Envolver el contrafractual
        exp = LocalGraphCounterfactualExplanation(context=self.context, dataset=self.dataset, oracle=self.oracle, explainer=self, input_instance=instance, counterfactual_instances=[cf_instance])

        return exp