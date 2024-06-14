import time

from src.evaluation.stages.metric_stage import MetricStage
from src.explanation.base import Explanation


class RuntimeStage(MetricStage):

    def process(self, explanation: Explanation) -> Explanation:
        explainer = explanation.explainer
        # Creating the counterfactual explanation and calculating the runtime
        start_time = time.time()
        counterfactual_exp = explainer.explain(explanation.input_instance)
        metic_value = time.time() - start_time

        # Writing the runtime info in the explanation
        self.write_into_explanation(counterfactual_exp, metic_value)
  
        # returning the explanation
        return counterfactual_exp