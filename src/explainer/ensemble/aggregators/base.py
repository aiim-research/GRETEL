from typing import List

from src.core.oracle_base import Oracle
from src.dataset.dataset_base import Dataset
from src.core.configurable import Configurable
from src.dataset.instances.base import DataInstance

from src.utils.cfg_utils import retake_oracle, retake_dataset

class ExplanationAggregator(Configurable):

    def init(self):
        super().init()
        self.dataset: Dataset = retake_dataset(self.local_config)
        self.oracle: Oracle = retake_oracle(self.local_config)

    def aggregate(self, instance: DataInstance, explanations: List[DataInstance]):
        pass