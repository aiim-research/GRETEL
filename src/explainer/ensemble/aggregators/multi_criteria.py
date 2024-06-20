from typing import List

import numpy as np

from src.core.factory_base import get_instance_kvargs
from src.explainer.ensemble.aggregators.base import ExplanationAggregator
from src.explainer.ensemble.aggregators.criterias.base_criteria import BaseCriteria
from src.explainer.ensemble.aggregators.distances.base_distance import BaseDistance
from src.explanation.local.graph_counterfactual import (
    LocalGraphCounterfactualExplanation,
)
from src.utils.cfg_utils import init_dflts_to_of


class ExplanationMultiCriteriaAggregator(ExplanationAggregator):
    def check_configuration(self):
        super().check_configuration()
        default_distance = "src.explainer.ensemble.aggregators.distances.euclidean_distance.EuclideanDistance"
        init_dflts_to_of(self.local_config, "distance", default_distance)

    def init(self):
        super().init()
        self.criterias: List[BaseCriteria] = [
            get_instance_kvargs(
                exp["class"], {"context": self.context, "local_config": exp}
            )
            for exp in self.local_config["parameters"]["criterias"]
        ]
        self.distance: BaseDistance = get_instance_kvargs(
            self.local_config["parameters"]["distance"]["class"],
            self.local_config["parameters"]["distance"]["parameters"],
        )

    def real_aggregate(
        self,
        explanations: List[LocalGraphCounterfactualExplanation],
    ) -> LocalGraphCounterfactualExplanation:
        input_inst = explanations[0].input_instance
        cf_instances = [
            cf for exp in explanations for cf in exp.counterfactual_instances
        ]
        criteria_matrix = (
            np.array(
                [
                    [criteria.evaluate(input_inst, cf) for criteria in self.criterias]
                    for cf in cf_instances
                ]
            ),
            np.array([criteria.gain_direction().value for criteria in self.criterias]),
        )
        criteria_matrix_normalized = self.__min_max_normalize(criteria_matrix)
        gain_directions = np.array(
            [criteria.gain_direction().value for criteria in self.criterias]
        )
        best_index = self.__find_best(
            criteria_matrix_normalized,
            gain_directions,
        )
        best_cf = cf_instances[best_index]
        return LocalGraphCounterfactualExplanation(
            context=self.context,
            dataset=self.dataset,
            oracle=self.oracle,
            explainer=None,  # Will be added later by the ensemble
            input_instance=input_inst,
            counterfactual_instances=[best_cf],
        )

    def __min_max_normalize(self, criteria_matrix: np.ndarray) -> np.ndarray:
        min_vals = np.min(criteria_matrix, axis=0)
        max_vals = np.max(criteria_matrix, axis=0)
        normalized_matrix = (criteria_matrix - min_vals) / (max_vals - min_vals)
        return normalized_matrix

    def __find_non_dominated_rows(
        self,
        criteria_matrix: np.ndarray,
        gain_directions: np.ndarray,
    ) -> np.ndarray:
        criteria_matrix_normalized = criteria_matrix * gain_directions
        num_rows = criteria_matrix.shape[0]
        non_dominated_indices = []
        for i in range(num_rows):
            dominated = False
            for j in range(num_rows):
                row1 = criteria_matrix_normalized[i]
                row2 = criteria_matrix_normalized[j]
                if i != j and np.all(row2 >= row1) and np.any(row2 > row1):
                    dominated = True
                    break
            if not dominated:
                non_dominated_indices.append(i)
        return np.array(non_dominated_indices)

    def __compute_ideal_point(
        self,
        criteria_matrix: np.ndarray,
        non_dominated_indices: np.ndarray,
        gain_directions: np.ndarray,
    ) -> np.ndarray:
        non_dominated_matrix = criteria_matrix[non_dominated_indices]
        ideal_point = (
            np.max(non_dominated_matrix * gain_directions, axis=0) * gain_directions
        )
        return ideal_point

    def __find_best(
        self,
        criteria_matrix: np.ndarray,
        gain_directions: np.ndarray,
    ) -> int:
        non_dominated_indices = self.__find_non_dominated_rows(
            criteria_matrix,
            gain_directions,
        )
        ideal_point = self.__compute_ideal_point(
            criteria_matrix,
            non_dominated_indices,
            gain_directions,
        )
        distances = self.distance.calculate(
            criteria_matrix[non_dominated_indices],
            ideal_point,
        )
        best_index = non_dominated_indices[np.argmin(distances)]
        return best_index
