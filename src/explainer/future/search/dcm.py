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
        super().init()
    
    def real_fit(self):
        super().real_fit()

    def fit(self):
        # leer para no realizar el entrenamiento si ya existe
        super().fit()
        #super().read()
        super().load_or_create()
        
        print('aquiiiiii')

        #if self.model:
        #   return self.model
        
        # Realizar el entrenamiento para guardarlo en cache

        memoids = None

        # Categorizar todos los grafos del dataset
        categorized_graph = [(self.oracle.predict(graph), graph) for graph in self.dataset]
        
        # Agrupar los grafos por categoría
        graphs_by_category = {}
        for category, graph in categorized_graph:
            if category not in graphs_by_category:
                graphs_by_category[category] = []
            graphs_by_category[category].append(graph)
        
        # Calcular el memoide de cada categoría
        memoids = {}
        for category, graphs in graphs_by_category.items():
                features = np.array([graph.data for graph in graphs])
                memoid = np.mean(features, axis=0)
                memoids[category] = memoid
        
        # Guardar los memoides
        self.model = memoids
        #super().write()
    
    def explain(self, instance):
        
        # Obtener la categoria de la instancia
        category = self.oracle.predict(instance)
        

        # Moverse por cada una de las categorias existentes en el dataset diferentes de la categoria actual, y comparar los memoides de cada una de esas categorias con la instancia actual, devolver el mas cercano
        min_distance = float('inf')
        closest_memoid = None
        for other_category, memoid in self._memoids.items():
            if other_category != category:
                distance = GraphEditDistanceMetric.compute(instance.data, memoid)
                if distance < min_distance:
                    min_distance = distance
                    closest_memoid = memoid       

        # Crear una instancia de grafo con el memoid mas cercano
        cf_instance = GraphInstance(id=instance.id, label=instance.label, data=closest_memoid, node_features=instance.node_features)

        # Envolver el contrafractual
        exp = LocalGraphCounterfactualExplanation(context=self.context, dataset=self.dataset, oracle=self.oracle, explainer=self, input_instance=instance, counterfactual_instances=[cf_instance])

        return exp