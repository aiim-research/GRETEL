from abc import abstractmethod, ABC
import numpy as np

from src.explanation.base import Explanation
from src.core.configurable import Configurable
from src.evaluation.stages.stage import Stage

class MeasurementStage(Stage):

    @abstractmethod
    def process(self, explanation: Explanation) -> Explanation:
        pass


    @classmethod
    def aggregate(cls, measure_list, instances_correctness_list=None):
        return np.mean(measure_list),np.std(measure_list)