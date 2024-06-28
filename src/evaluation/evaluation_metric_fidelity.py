from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer
from src.utils.metrics.fidelity import fidelity_metric


class FidelityMetric(EvaluationMetric):
    """As correctness measures if the algorithm is producing counterfactuals, but in Fidelity measures how faithful they are to the original problem,
     not just to the problem learned by the oracle. Requires a ground truth in the dataset
    """

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Fidelity'

    def evaluate(self, instance, explanation, oracle: Oracle=None, explainer: Explainer=None, dataset=None):
        return fidelity_metric(instance, explanation.top, oracle)
