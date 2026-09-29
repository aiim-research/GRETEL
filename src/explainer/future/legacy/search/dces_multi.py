import sys
from src.core.explainer_base import Explainer
from src.explainer.future.utils.explainer_transform import ExplainerTransformMeta
from src.explainer.search.dces import DCESExplainer as DCESExplainerOld
from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.utils.metrics.ged import GraphEditDistanceMetric

class DCESMulti(Explainer):
    def check_configuration(self):
        super().check_configuration()

    def init(self):
        super().init()
        self.distance_metric = GraphEditDistanceMetric()  
                
    def explain(self, instance):
        input_label = self.oracle.predict(instance)

        ctfs = []
        
        for ctf_candidate in self.dataset.instances:
            candidate_label = self.oracle.predict(ctf_candidate)

            if input_label != candidate_label:
                ctfs.append(ctf_candidate)

        ctfs_sorted = sorted(ctfs, key=lambda ctf: self.distance_metric.evaluate(instance, ctf, self.oracle))
        
        explanation = LocalGraphCounterfactualExplanation(context=self.context,
                                                                    dataset=self.dataset,
                                                                    oracle=self.oracle,
                                                                    explainer=self,
                                                                    input_instance=instance,
                                                                    counterfactual_instances=ctfs_sorted)
        
        return explanation