from typing import List

from src.explainer.ensemble.aggregators.base import ExplanationAggregator
from src.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation


class ExplanationMultiCriteriaAggregator(ExplanationAggregator):
    def real_aggregate(self, explanations: List[LocalGraphCounterfactualExplanation]) -> LocalGraphCounterfactualExplanation:
        pass
