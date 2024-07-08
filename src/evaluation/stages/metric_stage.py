from abc import abstractmethod
import numpy as np

from src.future.explanation.base import Explanation
from src.evaluation.stages.stage import Stage

class MetricStage(Stage):

    @abstractmethod
    def process(self, explanation: Explanation) -> Explanation:
        pass


    @classmethod
    def aggregate(cls, measure_list, instances_correctness_list=None):
        return np.mean(measure_list),np.std(measure_list)