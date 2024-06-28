from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer
from src.utils.metrics.ged import graph_edit_distance_metric

class SparsityMetric(EvaluationMetric):
    """Provides the ratio between the number of features modified to obtain the counterfactual example
     and the number of features in the original instance. Only considers structural features.
    """

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Sparsity'

    def evaluate(self, instance , explanation , oracle : Oracle=None, explainer : Explainer=None, dataset = None):
        instance_2 = explanation.top

        return graph_edit_distance_metric(instance.data, instance_2.data, instance.directed and instance_2.directed) / self.number_of_structural_features(instance)

    def number_of_structural_features(self, data_instance) -> float:
        return len(data_instance.get_nx().edges) + len(data_instance.get_nx().nodes)

