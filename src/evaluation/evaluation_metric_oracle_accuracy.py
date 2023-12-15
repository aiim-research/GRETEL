from src.evaluation.evaluation_metric_base import EvaluationMetric
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer


class OracleAccuracyMetric(EvaluationMetric):
    """As correctness measures if the algorithm is producing counterfactuals, but in Fidelity measures how faithful they are to the original problem,
     not just to the problem learned by the oracle. Requires a ground truth in the dataset
    """

    def __init__(self, config_dict=None) -> None:
        super().__init__(config_dict)
        self._name = 'Oracle_Accuracy'

    def evaluate(self, instance_1 , instance_2 , oracle : Oracle=None, explainer : Explainer=None, dataset = None):

        predicted_label_instance_1 = oracle.predict(instance_1)
        oracle._call_counter -= 1
        real_label_instance_1 = instance_1.label

        result = 1 if (predicted_label_instance_1 == real_label_instance_1) else 0
        
        return result