import time

from src.evaluation.stages.stage import Stage
from src.explanation.base import Explanation


class RuntimeStage(Stage):

    def process(self, exp: Explanation) -> Explanation:
        explainer = exp.explainer

        start_time = time.time()
        counterfactual_exp = explainer.explain(exp.input_instance)
        end_time = time.time()

        counterfactual_exp._stages_info[self.__class__.name] = (end_time - start_time)
  
        return counterfactual_exp