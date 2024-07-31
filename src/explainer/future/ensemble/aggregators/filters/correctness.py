from typing import List

from src.future.explanation.local.graph_counterfactual import LocalGraphCounterfactualExplanation
from src.explainer.future.ensemble.aggregators.filters.base import ExplanationFilter

class CorrectnessFilter(ExplanationFilter):   
    
    def real_filter(self, explanations: List[LocalGraphCounterfactualExplanation]) -> List[LocalGraphCounterfactualExplanation]:
        # Getting a list of all the explanations where invalid counterfactual examples where removed from the explanation
        # If the explanation does not contain any correct counterfactual it is considered invalid
        verified_explanations = [self._filter_explanation(exp) for exp in explanations]
        # Removing all invalid explanations from the list
        filtered_explanations = [exp for (exp_validity, exp) in verified_explanations if exp_validity]

        # Returning the list of valid explanations
        return filtered_explanations
    

    def _filter_explanation(self, explanation: LocalGraphCounterfactualExplanation):
        # Filtering the counterfactual instances of the explanation
        org_instance_label = self.oracle.predict(explanation.input_instance)
        filtered_cf_instances = [cf_instance for cf_instance in explanation.counterfactual_instances if self.oracle.predict(cf_instance) != org_instance_label]

        # Check if the explanation contains any correct counterfactual instance
        if len(filtered_cf_instances) > 0:
            filtered_exp = LocalGraphCounterfactualExplanation(context=explanation.context,
                                                            dataset=explanation.dataset,
                                                            oracle=explanation.oracle,
                                                            explainer=explanation.explainer,
                                                            input_instance=explanation.input_instance,
                                                            counterfactual_instances=filtered_cf_instances)
            
            # TODO this should be done automatically for all future attributes of an explanation
            filtered_exp.stages_info = explanation.stages_info
            filtered_exp.info = explanation.info
            # returns is_valid_explanation, new_explanation
            return True, filtered_exp

        # returns False is the new explanation does not contain any correct counterfactual instance
        return False, None
    
