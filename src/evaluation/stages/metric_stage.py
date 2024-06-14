from abc import abstractmethod
import numpy as np

from src.explanation.base import Explanation
from src.evaluation.stages.stage import Stage

class MetricStage(Stage):

    @abstractmethod
    def process(self, explanation: Explanation) -> Explanation:
        pass


    @classmethod
    def aggregate(cls, measure_list, instances_correctness_list=None):
        return np.mean(measure_list),np.std(measure_list)

    
    def write_into_explanation(self, exp: Explanation, value):
        exp._metrics_info[self.__class__.name] = value