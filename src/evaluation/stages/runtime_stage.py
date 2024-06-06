import time

from src.evaluation.stages.stage import Stage
from src.evaluation.stages.measurement_stage import MeasurementStage
from src.explanation.base import Explanation


class RuntimeStage(MeasurementStage):

    def process(self, explanation: Explanation) -> Explanation:
        explainer = explanation.explainer

        start_time = time.time()
        counterfactual_exp = explainer.explain(explanation.input_instance)
        end_time = time.time()

        counterfactual_exp._stages_info[self.__class__.name] = (end_time - start_time)
  
        return counterfactual_exp