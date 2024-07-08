from src.core.explainer_base import Explainer
from src.core.oracle_base import Oracle
from src.dataset.dataset_base import Dataset
from src.dataset.instances.graph import GraphInstance
from src.explainer.future.ensemble.aggregators.multi_criteria.criterias.base_criteria import (
    BaseCriteria,
)
from src.explainer.future.ensemble.aggregators.multi_criteria.criterias.gain_direction import (
    GainDirection,
)
from src.utils.metrics.ged import graph_edit_distance_metric


class GraphEditDistanceCriteria(BaseCriteria[GraphInstance]):
    def gain_direction(self):
        return GainDirection.MINIMIZE

    def calculate(
        self,
        first_instance: GraphInstance,
        second_instance: GraphInstance,
        oracle: Oracle,
        explainer: Explainer,
        dataset: Dataset,
    ) -> float:
        return graph_edit_distance_metric(
            first_instance.data,
            second_instance.data,
            first_instance.directed and second_instance.directed)
