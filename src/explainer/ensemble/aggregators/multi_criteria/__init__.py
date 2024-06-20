from typing import List

import numpy as np

from src.core.factory_base import get_instance_kvargs
from src.explainer.ensemble.aggregators.base import ExplanationAggregator
from src.explainer.ensemble.aggregators.multi_criteria.algorithm import find_best
from src.explainer.ensemble.aggregators.multi_criteria.criterias.base_criteria import (
    BaseCriteria,
)
from src.explainer.ensemble.aggregators.multi_criteria.distances.base_distance import (
    BaseDistance,
)
from src.explanation.local.graph_counterfactual import (
    LocalGraphCounterfactualExplanation,
)
from src.utils.cfg_utils import init_dflts_to_of


class ExplanationMultiCriteriaAggregator(ExplanationAggregator):
    def check_configuration(self):
        super().check_configuration()
        default_distance = "src.explainer.ensemble.aggregators.multi_criteria.distances.euclidean_distance.EuclideanDistance"
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
            {
                "context": self.context,
                "local_config": self.local_config["parameters"]["distance"],
            },
        )

    def real_aggregate(
        self,
        explanations: List[LocalGraphCounterfactualExplanation],
    ) -> LocalGraphCounterfactualExplanation:
        input_inst = explanations[0].input_instance
        cf_instances = [
            cf for exp in explanations for cf in exp.counterfactual_instances
        ]
        criteria_matrix = np.array(
            [
                [criteria.calculate(input_inst, cf) for criteria in self.criterias]
                for cf in cf_instances
            ]
        )
        gain_directions = np.array(
            [criteria.gain_direction().value for criteria in self.criterias]
        )
        best_index = find_best(
            criteria_matrix,
            gain_directions,
            self.distance.calculate,
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
