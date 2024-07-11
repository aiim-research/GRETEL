from typing import TYPE_CHECKING

from src.future.explanation.local.graph_counterfactual import (
    LocalGraphCounterfactualExplanation,
)

if TYPE_CHECKING:
    from src.core.explainer_base import Explainer
    from src.dataset.instances.graph import GraphInstance


class ExplainerTransformMeta(type):
    def __new__(cls, name, bases, dct):
        original_explain = dct.get("explain", None)
        if original_explain:

            def new_explain(self: Explainer, instance: GraphInstance):
                cf_instance = original_explain(self, instance)
                return LocalGraphCounterfactualExplanation(
                    context=self.context,
                    dataset=self.dataset,
                    oracle=self.oracle,
                    explainer=self,
                    input_instance=instance,
                    counterfactual_instances=[cf_instance],
                )

            dct["explain"] = new_explain
        return super().__new__(cls, name, bases, dct)
