from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric


class SparsityMetric(EvaluationMetric):
    """Provides the ratio between the number of features modified to obtain the counterfactual example
     and the number of features in the original instance. Only considers structural features.
    """

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Sparsity'

    def evaluate(self, instance_1 , instance_2 , oracle : Oracle=None, explainer : Explainer=None, dataset = None):
        ged = GraphEditDistanceMetric()
        return ged.evaluate(instance_1, instance_2, oracle)/self.number_of_structural_features(instance_1)

    def number_of_structural_features(self, data_instance) -> float:
        return len(data_instance.get_nx().edges) + len(data_instance.get_nx().nodes)
