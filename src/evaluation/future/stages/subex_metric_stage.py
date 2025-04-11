from abc import abstractmethod
import numpy as np

from src.future.explanation.base import Explanation
from src.evaluation.future.stages.stage import Stage

class SubexMetricStage(Stage):

    def check_configuration(self):
        super().check_configuration()

        if 'target' not in self.local_config['parameters']:
            raise Exception('A SubExMetricStage requires a target subexplanation')

    def init(self):
        super().init()
        self.target = self.local_config['parameters']['target']

    @abstractmethod
    def process(self, explanation: Explanation) -> Explanation:
        pass


    @classmethod
    def aggregate(cls, measure_list, instances_correctness_list=None):
        return np.mean(measure_list),np.std(measure_list)