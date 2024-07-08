from src.core.explainer_base import Explainer
from src.core.oracle_base import Oracle
from src.dataset.instances.graph import GraphInstance


class FidelityMetric:
    def evaluate(
        self,
        instance_1: GraphInstance,
        instance_2: GraphInstance,
        oracle: Oracle=None,
        explainer: Explainer=None,
        dataset=None,
    ) -> float:
        return fidelity_metric(instance_1, instance_2, oracle)


def fidelity_metric(
    instance_1: GraphInstance,
    instance_2: GraphInstance,
    oracle : Oracle,
) -> float:
    label_instance_1 = oracle.predict(instance_1)
    label_instance_2 = oracle.predict(instance_2)
    oracle._call_counter -= 2
    return fidelity_metric_with_predictions(
        instance_1,
        label_instance_1,
        label_instance_2,
    )


def fidelity_metric_with_predictions(
    instance_1: GraphInstance,
    label_instance_1,
    label_instance_2,
) -> float:
    prediction_fidelity = 1 if (label_instance_1 == instance_1.label) else 0
    counterfactual_fidelity = 1 if (label_instance_2 == instance_1.label) else 0
    result = prediction_fidelity - counterfactual_fidelity
    return result
